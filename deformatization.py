# For each format spec, we have a set of generations, a set of embeddings and an accuracy score.
# What if we take all the embeddings and run PCA on them?
# We are not guaranteed that formatting differences will be captured in top principal components,
# as they describe the variation within the dataset.

# What if we take differences between embeddings of generations with different formatting?
# We can then run PCA on these difference vectors.
# The top principal components will then hopefullydescribe the formatting differences.

# Let's use the worst-performing format as a `target`, and all other formats as `baseline`.
# With 10 formats total, for each sample in the dataset there will be 9 difference vectors.
# We can then orthogonalize the model weights with respect to top principal components.
# This might remove the sensitivity to formatting.

# Alternatively, we could use best-performing format as a `target`. Or a random format.

import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import List
from main import _argmin, _argmax, select_few_shot_examples
from abliteration import orthogonalize_model_weights
from format_utils import build_gsm8k_few_shot_prompt, FormatSpecification

def build_differences_tensor(embeddings_list: List[torch.Tensor], n_blocks: int, block_size: int) -> torch.Tensor:
    n_format_specs = len(embeddings_list)
    n_samples = embeddings_list[0].shape[0]

    differences = []
    for _ in range(n_blocks):
        random_sample_indices = torch.randint(0, n_samples, (block_size,))
        first_format_idx, second_format_idx = torch.randperm(n_format_specs)[:2]
        first_embeddings = embeddings_list[first_format_idx][random_sample_indices]
        second_embeddings = embeddings_list[second_format_idx][random_sample_indices]
        diffs = first_embeddings - second_embeddings
        norms = torch.linalg.norm(diffs, dim=-1, keepdim=True)
        print(f"{len(norms)=}")
        print("Norms:", norms.float().cpu().numpy())
        debug_diffs = first_embeddings - embeddings_list[second_format_idx][:block_size]
        debug_norms = torch.linalg.norm(debug_diffs, dim=-1, keepdim=True)
        print("Debug norms:", debug_norms.float().cpu().numpy())
        diffs = diffs / (norms + 1e-8)
        differences.append(diffs)
        
    return torch.cat(differences, dim=0)

def estimate_format_directions(
        full_experiment_path: str,
        layer_index: int,
        n_format_specs: int,
        n_principal_components: int,
        print_debug_info: bool,
        seed: int,
    ):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # metrics_path = os.path.join(full_experiment_path, "metrics.json")
    # if not os.path.exists(metrics_path):
    #     raise FileNotFoundError(f"Metrics not found at {metrics_path}")
    # metrics = json.load(open(metrics_path))

    embeddings_list = []
    for format_index in range(n_format_specs):
        embeddings_path = os.path.join(full_experiment_path, f"embeddings_format_spec_{format_index}.pth")
        if not os.path.exists(embeddings_path):
            print(f"Embeddings not found for format spec {format_index}, skipping...")
            continue

        embeddings = torch.load(embeddings_path)
        embeddings_list.append(embeddings[:, layer_index, :])
    
    differences_tensor = build_differences_tensor(embeddings_list, n_blocks=128, block_size=32)
    differences_tensor = differences_tensor.float()
    print(f"{differences_tensor.shape=}")
    # [n_format_specs - 1 * n_samples, n_dimensions]
    mean_diff = torch.mean(differences_tensor, dim=0)
    # differences_tensor = differences_tensor - mean_diff[None]

    U, S, Vh = torch.linalg.svd(differences_tensor, full_matrices=False)
    print(U.shape, S.shape, Vh.shape)
    # [n_format_specs - 1 * n_samples, n_dimensions]
    # min(n_format_specs - 1 * n_samples, n_dimensions)
    # [n_dimensions, n_dimensions]

    top_principal_components = Vh[:n_principal_components, :]
    assert top_principal_components.shape == (n_principal_components, embeddings_list[0].shape[-1]), \
        f"{top_principal_components.shape=} != {(n_principal_components, embeddings_list[0].shape[-1])=}"

    first_pca_format_direction = top_principal_components[0]

    # Align the first format direction with the mean difference vector
    if torch.cosine_similarity(first_pca_format_direction, mean_diff, dim=-1) < 0:
        top_principal_components[0] = -first_pca_format_direction

    if not print_debug_info:
        return top_principal_components

    variances = S**2
    total_variance = torch.sum(variances)
    variance_ratios = variances / total_variance
    cumulative_variance_ratios = torch.cumsum(variance_ratios, dim=0)
    for i, ratio in enumerate(cumulative_variance_ratios[:n_principal_components + 5]):
        print(f"PC {i}: {ratio:.3f}")

    first_pca_format_direction = top_principal_components[0]
    second_pca_format_direction = top_principal_components[1]
    third_pca_format_direction = top_principal_components[2]

    # Stack vectors into a matrix and compute pairwise cosine similarities
    vectors = torch.stack([mean_diff, first_pca_format_direction, second_pca_format_direction, third_pca_format_direction])
    similarity_matrix = torch.nn.functional.cosine_similarity(
        vectors.unsqueeze(1), 
        vectors.unsqueeze(0), 
        dim=2
    )
    
    # Print similarity matrix with rounded values
    print("\nCosine similarity matrix:")
    print("                        mean_diff  first_pca  second_pca  third_pca")
    labels = ["mean_diff", "first_pca", "second_pca", "third_pca"]
    for i, label in enumerate(labels):
        row = [f"{similarity_matrix[i,j]:.3f}" for j in range(len(labels))]
        print(f"{label:12} {' '.join(row)}")

    return top_principal_components

def create_prompt():
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    few_shot_examples = select_few_shot_examples(train_dataset, 24, 100, print_debug_info=False)
    test_example = {"question": "Alice has 20 quarters. She wants to exchange them for nickels and so she goes to the bank. After getting back from the bank, she discovers that 20% of the nickels are iron nickels worth $3 each. What is the total value of her money now?"}
    format_spec = FormatSpecification(
        descriptor_transformation=lambda x: x.upper(),
        descriptor_transformation_str="lambda x: x.upper()",
        separator=":: ",
        space="; \n",
        first_descriptor="question",
        second_descriptor="reasoning",
        third_descriptor="answer",
    )
    reasoning_answer_separator = "####"
    prompt = build_gsm8k_few_shot_prompt(test_example, few_shot_examples, format_spec, reasoning_answer_separator)
    return prompt

@torch.inference_mode()
def generate_from_prompt(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_attention_mask=True, return_tensors="pt", truncation=True)
    outputs = model.generate(
        input_ids=inputs.input_ids.to(model.device),
        attention_mask=inputs.attention_mask.to(model.device),
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=["Question", "question", "QUESTION", "You are an AI assistant"],
        tokenizer=tokenizer
    )
    outputs = outputs[:, len(inputs.input_ids[0]):]
    generation = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name-or-path", type=str, required=True)
    parser.add_argument("-e", "--full-experiment-path", type=str, required=True)
    parser.add_argument("-n", "--n-principal-components", type=int, required=True)
    parser.add_argument("-l", "--layer-index", type=int, required=True)
    parser.add_argument("--n-format-specs", type=int, default=10)
    parser.add_argument("--cache-dir", type=str, default="../.cache/huggingface/hub")
    parser.add_argument("--seed", type=int, default=24)
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    torch_dtype = torch.bfloat16
    attn_implementation = "eager"
    device = "cuda:0"

    format_directions = estimate_format_directions(
        args.full_experiment_path,
        args.layer_index,
        args.n_format_specs,
        args.n_principal_components,
        print_debug_info=True,
        seed=args.seed,
    ).to(torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        cache_dir=args.cache_dir,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    prompt = create_prompt()
    print("Example prompt:\n\n")
    print(prompt)
    print("=" * 100)

    generation = generate_from_prompt(model, tokenizer, prompt)
    print("Original model generation:\n\n")
    print(generation)
    print("=" * 100)

    for i in range(args.n_principal_components):
        stats = orthogonalize_model_weights(model, format_directions[i], 1.0)
        print(f"Deformatted model, stage {i}:")
        generation = generate_from_prompt(model, tokenizer, prompt)
        print(generation)
        print("=" * 100)

    path = args.model_name_or_path.split("/")[-1] + f"-deformatted-pca-{args.n_principal_components}-layer-{args.layer_index}"
    model.save_pretrained(args.cache_dir + "/" + path)
    tokenizer.save_pretrained(args.cache_dir + "/" + path)

if __name__ == "__main__":
    main()