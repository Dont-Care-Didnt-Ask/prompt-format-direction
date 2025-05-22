import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import Callable, List, Dict, Any, Tuple, Optional

def setup_pytorch_model_tokenizer(
    model_name_or_path: str,
    device_preference: Optional[str] = None,
    use_half_precision: bool = True,
    padding_side: str = "left"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Loads a Hugging Face model and tokenizer, and prepares them for inference.
    Sets tokenizer to use left padding.

    Args:
        model_name_or_path: Identifier for the Hugging Face model.
        device_preference: Preferred device ("cuda", "cpu"). Autodetects if None.
        use_half_precision: Whether to use half-precision (fp16/bf16) if on CUDA.

    Returns:
        A tuple containing the model, tokenizer, and torch device.
    """
    if device_preference:
        device = torch.device(device_preference)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name_or_path}: {e}")
        raise

    print(f"Original tokenizer padding side: {tokenizer.padding_side}")
    tokenizer.padding_side = padding_side
    print(f"Set tokenizer padding side to: {tokenizer.padding_side}")

    # Set pad token if not present
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            print(f"Tokenizer missing pad_token_id. Setting pad_token to eos_token ({tokenizer.eos_token}).")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            print("Tokenizer missing both pad_token_id and eos_token_id. Adding a new pad token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # Model embeddings might need resizing if a new token is added.

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    except Exception as e:
        print(f"Error loading model {model_name_or_path}: {e}")
        raise
    
    if tokenizer.pad_token == '[PAD]' and tokenizer.pad_token_id >= model.config.vocab_size:
         print(f"Resizing model token embeddings to accommodate new pad token. Old vocab size: {model.config.vocab_size}, new tokenizer vocab size: {len(tokenizer)}")
         model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.eval()

    if use_half_precision and device.type == "cuda":
        if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
            print("Using bfloat16 precision.")
            model = model.bfloat16()
        else:
            print("Using float16 precision.")
            model = model.half()
    
    print(f"Model {model_name_or_path} loaded on {device} with {tokenizer.padding_side} padding.")
    return model, tokenizer, device

# @torch.compile()
def get_last_token_embeddings(
    dataset: Dataset,
    prompt_builder_fn: Callable[[Dict[str, Any]], str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int = 8
) -> torch.Tensor:
    """
    Collects last token embeddings for each prompt from all transformer layers. Assumes tokenizer uses left padding.

    Args:
        dataset: Hugging Face Dataset.
        prompt_builder_fn: Function that takes a dataset example (dict) and returns a prompt string.
        model: Pre-loaded PyTorch AutoModelForCausalLM.
        tokenizer: Pre-loaded PyTorch AutoTokenizer.
        device: Torch device.
        batch_size: Batch size for processing.

    Returns:
        A tensor of shape (num_dataset_samples, num_hidden_layers, hidden_dimension).
    """
    all_embeddings_list = []
    
    if not hasattr(model.config, 'num_hidden_layers') or not hasattr(model.config, 'hidden_size'):
        raise ValueError("Model config does not have 'num_hidden_layers' or 'hidden_size'. Cannot determine embedding dimensions.")
        
    num_hidden_layers = model.config.num_hidden_layers
    hidden_dimension = model.config.hidden_size

    if tokenizer.padding_side != 'left':
        # Important check, as the logic below relies on it.
        raise ValueError("Tokenizer padding_side must be 'left' for get_last_token_embeddings.")

    # print(f"Extracting embeddings. Num layers: {num_hidden_layers}, Hidden dim: {hidden_dimension}.")

    for i in range(0, len(dataset), batch_size):
        batch_examples = dataset[i:i + batch_size]
        
        # The dataset __getitem__ for a slice returns a dict of lists.
        # We need to reconstruct individual examples to pass to prompt_builder_fn.
        # Assuming all lists in batch_examples have the same length (the current batch size).
        current_batch_size = len(batch_examples[list(batch_examples.keys())[0]])
        prompts_batch = []
        for j in range(current_batch_size):
            example = {key: batch_examples[key][j] for key in batch_examples}
            prompts_batch.append(prompt_builder_fn(example))

        try:
            inputs = tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=tokenizer.model_max_length if tokenizer.model_max_length else 2048, # Fallback max_length
                return_attention_mask=True
            )
        except Exception as e:
            print(f"Error during tokenization for batch starting at index {i}: {e}")
            raise

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            except Exception as e:
                print(f"Error during model inference for batch starting at index {i}: {e}")
                raise

        hidden_states_tuple = outputs.hidden_states 
        # hidden_states_tuple[0] is the input embeddings.
        # hidden_states_tuple[1:] are the outputs of each transformer layer.
        # So, len(hidden_states_tuple[1:]) == num_hidden_layers

        if not hidden_states_tuple or len(hidden_states_tuple) < num_hidden_layers + 1:
             raise ValueError(f"Expected at least {num_hidden_layers + 1} sets of hidden states, but got {len(hidden_states_tuple) if hidden_states_tuple else 0}.")

        # With left padding, the last *actual* token of the prompt is always at the last position
        # of the sequence length dimension of the hidden states, for every item in the batch.
        # The hidden_states tensors will have the same sequence length as input_ids.
        # sequence_length_in_batch = input_ids.shape[1] # Max sequence length in this batch

        # Extract all last token embeddings across layers in one operation
        last_token_embeddings = torch.stack([
            hidden_states_tuple[layer_idx + 1][:, -1, :] 
            for layer_idx in range(num_hidden_layers)
        ], dim=1)
        
        # Add each sample's embeddings to the list
        all_embeddings_list.extend([last_token_embeddings[j] for j in range(input_ids.size(0))])

    if not all_embeddings_list:
        # This could happen if dataset was empty or all samples resulted in errors/empty sequences.
        print("Warning: No embeddings were collected. Returning an empty tensor.")
        return torch.empty((0, num_hidden_layers, hidden_dimension), dtype=model.dtype, device=device)
        
    final_embeddings_tensor = torch.stack(all_embeddings_list, dim=0)
    return final_embeddings_tensor

# @torch.compile()
def get_generations(
    dataset: Dataset,
    prompt_builder_fn: Callable[[Dict[str, Any]], str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    stop_strings: List[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    new_column_name: str = "generated_text"
) -> List[str]:
    """
    Generates text continuations for prompts.

    Args:
        dataset: Hugging Face Dataset.
        prompt_builder_fn: Function that takes a dataset example (dict) and returns a prompt string.
        model: Pre-loaded PyTorch AutoModelForCausalLM.
        tokenizer: Pre-loaded PyTorch AutoTokenizer.
        device: Torch device.
        stop_strings: List of strings to stop generation when encountered.
        batch_size: Batch size for processing.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. 0.0 for greedy decoding.
        new_column_name: Name of the new column for generated text.

    Returns:
        ~~The input Dataset object, augmented with the new_column_name~~.
        A list of generated texts.
    """
    all_generations_list = []
    
    do_sample = False
    if temperature > 0.0:
        do_sample = True
    elif temperature < 0.0:
        raise ValueError("Temperature must be non-negative.")

    if tokenizer.padding_side != 'left':
        # This function's generation extraction logic relies on left padding.
        raise ValueError("Tokenizer padding_side must be 'left' for get_generations.")

    # print(f"Generating text. Max new tokens: {max_new_tokens}, Temperature: {temperature}, Do sample: {do_sample}. Using left padding.")

    for i in range(0, len(dataset), batch_size):
        batch_examples = dataset[i:i + batch_size]
        current_batch_size = len(batch_examples[list(batch_examples.keys())[0]])
        prompts_batch = []
        for j in range(current_batch_size):
            example = {key: batch_examples[key][j] for key in batch_examples}
            prompts_batch.append(prompt_builder_fn(example))

        try:
            inputs = tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=tokenizer.model_max_length if tokenizer.model_max_length else 2048,
                return_attention_mask=True
            )
        except Exception as e:
            print(f"Error during tokenization for batch starting at index {i}: {e}")
            raise
            
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # This is the length of the (left-)padded input sequence
        num_prompt_tokens_padded_length = input_ids.shape[1]

        with torch.no_grad():
            try:
                generated_sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer
                )
            except Exception as e:
                print(f"Error during model generation for batch starting at index {i}: {e}")
                raise

        # With left padding, input_ids are like [P, P, T1, T2, T3]
        # generated_sequences will be [P, P, T1, T2, T3, G1, G2, G3]
        # The original number of tokens *including padding* was num_prompt_tokens_padded_length.
        # So, we want to slice from that point onwards.
        generated_tokens_only = generated_sequences[:, num_prompt_tokens_padded_length:]
        
        try:
            decoded_texts = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)
        except Exception as e:
            print(f"Error during decoding for batch starting at index {i}: {e}")
            raise

        all_generations_list.extend(decoded_texts)

    return all_generations_list
    # try:
    #     updated_dataset = dataset.add_column(name=new_column_name, column=all_generations_list)
    # except Exception as e:
    #     print(f"Error adding column '{new_column_name}' to dataset: {e}")
    #     print(f"{len(all_generations_list)=}, {len(dataset)=}")
    #     # This might happen if len(all_generations_list) != len(dataset) due to an error.
    #     # Or if the dataset is an iterable dataset not supporting add_column directly (though input type is Dataset).
    #     raise
        
    # return updated_dataset

if __name__ == '__main__':
    pass