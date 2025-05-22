import os
import random
from typing import List
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)
from datasets import load_dataset

def load_instructions(dataset_id, column, n_instructions):
    dataset = load_dataset(dataset_id, split="train")
    indices = random.sample(range(len(dataset)), n_instructions * 2)
    return [dataset[i][column] for i in indices[:n_instructions]], [
        dataset[i][column] for i in indices[n_instructions:]
    ]

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.5,
        min_p=0.1,
        repetition_penalty=1.05,
        streamer=TextStreamer(tokenizer),
    )

def generate_outputs(model, tokenizer, instructions, system_prompt):
    inputs = [
        tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        for instruction in instructions
    ]

    outputs = [
        model.generate(
            input,
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )["hidden_states"][0]
        for input in tqdm(inputs, desc="Generating outputs")
    ]
    return outputs

def orthogonalize_matrix(matrix, vec, weight):
    vec = vec.view(-1).to(matrix.device)

    if matrix.shape[-1] == vec.shape[0]:
        proj = torch.einsum("...d,d->...", matrix, vec).unsqueeze(-1) * vec.unsqueeze(0)
        return matrix - weight * proj
    elif matrix.shape[0] == vec.shape[0]:
        proj = torch.einsum("d...,d->...", matrix, vec).unsqueeze(0) * vec.unsqueeze(-1)
        return matrix - weight * proj
    else:
        raise ValueError(
            f"Matrix shape {matrix.shape} incompatible with vector shape {vec.shape}"
        )

def calculate_refusal_direction(target_outputs, baseline_outputs, layer_idx):
    # Extract hidden states from outputs
    target_hidden = [output[layer_idx][:, -1, :] for output in target_outputs]
    baseline_hidden = [output[layer_idx][:, -1, :] for output in baseline_outputs]

    # Calculate refusal direction
    target_mean = torch.stack(target_hidden).mean(dim=0)
    baseline_mean = torch.stack(baseline_hidden).mean(dim=0)
    refusal_dir = target_mean - baseline_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    print(f"{target_mean.shape=}, {baseline_mean.shape=}, {refusal_dir.shape=}")

    return refusal_dir

def calculate_refusal_directions_pca(target_outputs, baseline_outputs, layer_idx, n_pairs, n_directions, random_seed) -> torch.Tensor:
    """
    Calculates the refusal directions using PCA on the differences between
    target and baseline hidden states. Uses a local random generator for reproducibility.
    Prints cumulative explained variance for top components.

    Returns a tensor of shape [n_directions, hidden_dim] -- top-n_directions principal components.
    """
    # 1. Random Seed Handling & Device Setup
    # Use a local generator for reproducibility without affecting global state
    # Hidden states are on some device, generator should be CPU for randint, which is default
    device = target_outputs[0][0].device # Assuming target_outputs is not empty and structure is consistent
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    # 2. Hidden State Extraction & Preparation
    # target_hidden/baseline_hidden are lists of tensors, each [1, hidden_dim]
    target_hidden_list = [output[layer_idx][:, -1, :] for output in target_outputs]
    baseline_hidden_list = [output[layer_idx][:, -1, :] for output in baseline_outputs]

    if not target_hidden_list or not baseline_hidden_list:
        raise ValueError("Target or baseline hidden states list is empty.")

    # Concatenate into matrices: [num_samples, hidden_dim]
    target_hidden_matrix = torch.cat(target_hidden_list, dim=0).float()
    baseline_hidden_matrix = torch.cat(baseline_hidden_list, dim=0).float()

    num_target_samples = target_hidden_matrix.shape[0]
    num_baseline_samples = baseline_hidden_matrix.shape[0]

    if num_target_samples == 0 or num_baseline_samples == 0:
        raise ValueError("Not enough samples in target or baseline hidden states after processing.")

    # 3. Vectorized Difference Calculation & Normalization
    # Sample indices
    indices_target = torch.randint(0, num_target_samples, (n_pairs,), generator=generator)
    indices_baseline = torch.randint(0, num_baseline_samples, (n_pairs,), generator=generator)

    # Select vectors
    selected_targets = target_hidden_matrix[indices_target]
    selected_baselines = baseline_hidden_matrix[indices_baseline]

    # Compute differences
    diff_matrix = selected_targets - selected_baselines  # Shape: [n_pairs, hidden_dim]

    # Normalize row-wise
    norms = torch.linalg.norm(diff_matrix, dim=1, keepdim=True)
    epsilon = 1e-12  # Small epsilon to prevent division by zero
    # Rows with zero norm will remain zero vectors, which is fine for SVD.
    diff_matrix_normalized = diff_matrix / (norms + epsilon)
    
    # Handle cases where all diff_matrix_normalized rows might be zero if all diff_matrix rows were zero
    if not torch.any(diff_matrix_normalized.abs() > epsilon): # Check if matrix is effectively zero
         # This could happen if all target-baseline pairs were identical.
         # Depending on desired behavior, could raise error or return a zero vector.
         # For now, SVD will likely produce zero singular values.
         print("Warning: All normalized difference vectors are close to zero.")


    # 4. SVD and Principal Component
    mean_diff = diff_matrix_normalized.mean(dim=0, keepdim=True)
    diff_matrix_normalized = diff_matrix_normalized - mean_diff
    try:
        # U: (n_pairs, k), S: (k,), Vh: (k, hidden_dim) where k = min(n_pairs, hidden_dim)
        U, S, Vh = torch.linalg.svd(diff_matrix_normalized, full_matrices=False)
    except torch.linalg.LinAlgError as e:
        print(f"SVD failed: {e}")
        # Consider the shape of diff_matrix_normalized for debugging if SVD fails
        print(f"Shape of matrix fed to SVD: {diff_matrix_normalized.shape}")
        raise

    if Vh.shape[0] == 0: # No principal components found (e.g. n_pairs might be 0, or SVD issue)
        raise ValueError("SVD did not return any principal components (Vh is empty).")

    # Get mean_diff + top-20 principal components
    n_directions = min(n_directions, Vh.shape[0])
    refusal_dirs_pca = torch.cat(
        [mean_diff, Vh[:n_directions, :]],
        dim=0
    )
    
    # Re-normalize each direction -- SVD does not quite guarantee unit norm
    norms = torch.linalg.norm(refusal_dirs_pca, dim=1, keepdim=True)
    refusal_dirs_pca = refusal_dirs_pca / norms
    
    # Coherence check

    diff_mean_refusal_dir = calculate_refusal_direction(target_outputs, baseline_outputs, layer_idx).squeeze(0)
    mean_diff_refusal_dir = refusal_dirs_pca[0]
    first_pca_refusal_dir = refusal_dirs_pca[1]
    second_pca_refusal_dir = refusal_dirs_pca[2]

    # Stack vectors into a matrix and compute pairwise cosine similarities
    vectors = torch.stack([diff_mean_refusal_dir, mean_diff_refusal_dir, first_pca_refusal_dir, second_pca_refusal_dir])
    similarity_matrix = torch.nn.functional.cosine_similarity(
        vectors.unsqueeze(1), 
        vectors.unsqueeze(0), 
        dim=2
    )
    
    # Print similarity matrix with rounded values
    print("\nCosine similarity matrix:")
    print("                        diff_mean  mean_diff  first_pca  second_pca")
    labels = ["diff_mean", "mean_diff", "first_pca", "second_pca"]
    for i, label in enumerate(labels):
        row = [f"{similarity_matrix[i,j]:.3f}" for j in range(len(labels))]
        print(f"{label:12} {' '.join(row)}")


    # 5. Cumulative Explained Variance
    squared_singular_values = S**2
    total_variance = torch.sum(squared_singular_values)

    if total_variance < epsilon:
        print("Total variance is close to zero. Cannot compute meaningful explained variance.")
        return refusal_dirs_pca

    explained_variance_ratios = squared_singular_values / total_variance
    cumulative_explained_variance = torch.cumsum(explained_variance_ratios, dim=0)
    
    print("Cumulative explained variance by principal components:")
    for i in range(n_directions):
        print(f"  Top {i+1:2d} PC(s): {cumulative_explained_variance[i].item():.2f}")

    return refusal_dirs_pca.bfloat16()

def orthogonalize_model_weights(model, refusal_dir, refusal_weight):
    """Orthogonalize model weights with respect to the refusal direction.
    Caution! This function modifies the model in place.
    """
    refusal_dir = refusal_dir.view(-1).to(model.device)
    stats = {"embed_tokens": False, "attention_o_proj": 0, "mlp_proj": 0}

    # Embed tokens
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens.weight.data = orthogonalize_matrix(
            model.model.embed_tokens.weight.data, refusal_dir, refusal_weight
        )
        stats["embed_tokens"] = True

    # Layer projections
    for layer in tqdm(model.model.layers, desc="Orthogonalizing weights"):
        # Attention output projection
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            layer.self_attn.o_proj.weight.data = orthogonalize_matrix(
                layer.self_attn.o_proj.weight.data, refusal_dir, refusal_weight
            )
            stats["attention_o_proj"] += 1

        # MLP projection (down_proj or c_proj)
        if hasattr(layer, "mlp"):
            proj_name = (
                "down_proj"
                if hasattr(layer.mlp, "down_proj")
                else "c_proj"
                if hasattr(layer.mlp, "c_proj")
                else None
            )
            if proj_name:
                getattr(layer.mlp, proj_name).weight.data = orthogonalize_matrix(
                    getattr(layer.mlp, proj_name).weight.data, refusal_dir, refusal_weight
                )
                stats["mlp_proj"] += 1

    # Check if orthogonalization succeeded
    if (
        not stats["embed_tokens"]
        and stats["attention_o_proj"] == 0
        and stats["mlp_proj"] == 0
    ):
        raise RuntimeError(
            "Failed to orthogonalize any model weights. Model not abliterated."
        )

    return stats

@torch.inference_mode()
def main():
    # Configuration
    torch_dtype = torch.bfloat16
    attn_implementation = "eager"
    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    N_INSTRUCTIONS = 128
    TARGET_LAYER = 0.55
    REFUSAL_WEIGHT = 1
    PRIVATE_UPLOAD = True
    cache_dir = "../.cache/huggingface/hub"

    # Dataset configuration
    target_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    target_dataset = "mlabonne/harmful_behaviors"
    target_column = "text"
    baseline_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    baseline_dataset = "mlabonne/harmless_alpaca"
    baseline_column = "text"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    LAYER_IDX = int(TARGET_LAYER * len(model.model.layers))
    
    print("Before abliteration:")
    for i in range(3):
        print("=" * 50, "attempt", i + 1, "=" * 50)
        generate_response(model, tokenizer, "How to write a computer virus?")

    print("=" * 100)

    # Load instructions
    target_instructions, target_test = load_instructions(
        target_dataset, target_column, N_INSTRUCTIONS
    )
    baseline_instructions, baseline_test = load_instructions(
        baseline_dataset, baseline_column, N_INSTRUCTIONS
    )

    # Generate outputs
    baseline_outputs = generate_outputs(
        model, tokenizer, baseline_instructions, system_prompt=baseline_prompt
    )
    target_outputs = generate_outputs(model, tokenizer, target_instructions, system_prompt=target_prompt)

    # Calculate refusal direction
    # refusal_dir = calculate_refusal_direction(target_outputs, baseline_outputs, LAYER_IDX)
    refusal_dirs = calculate_refusal_directions_pca(
        target_outputs,
        baseline_outputs,
        LAYER_IDX,
        n_pairs=512,
        n_directions=20,
        random_seed=24
    )
    refusal_dir = refusal_dirs[0]
    del target_outputs, baseline_outputs

    # Orthogonalize model weights
    for i in range(2):
        print("Removing refusal direction", i + 1)
        refusal_dir = refusal_dirs[i]
        stats = orthogonalize_model_weights(model, refusal_dir, REFUSAL_WEIGHT)
        print(f"Orthogonalization stats: {stats}")

    del refusal_dirs

    print("After abliteration:")
    for i in range(3):
        print("=" * 50, "attempt", i + 1, "=" * 50)
        generate_response(model, tokenizer, "How to write a computer virus?")

    print("=" * 100)

    # Push the model
    path = MODEL_ID.split("/")[-1] + "-abliterated"
    model.save_pretrained(cache_dir + "/" + path)
    tokenizer.save_pretrained(cache_dir + "/" + path)
    print(f"Pushing model to {path}")
    # model.push_to_hub(path, private=PRIVATE_UPLOAD)
    # tokenizer.push_to_hub(path, private=PRIVATE_UPLOAD)

if __name__ == "__main__":
    main()