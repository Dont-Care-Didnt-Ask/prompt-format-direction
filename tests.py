import pytest
import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer
from typing import Dict, Any, Tuple

# Assuming pytorch_llm_utils.py is in the same directory or accessible in PYTHONPATH
from pytorch_llm_utils import (
    setup_pytorch_model_tokenizer,
    get_last_token_embeddings,
    get_generations
)

MODEL_ID = "gpt2"  # Using a small, standard model for tests

# 1. Prompt builder function (moved here)
def simple_prompt_builder(example: Dict[str, Any]) -> str:
    return f"Prompt: {example['text']} Continue this:"

# 2. Fixture for the dummy dataset
@pytest.fixture(scope="session")
def dummy_dataset() -> Dataset:
    dummy_data = {
        "id": [1, 2, 3],
        "text": [
            "This is a short sentence.",
            "This is a significantly longer sentence to test padding.",
            "A medium one."
        ]
    }
    return Dataset.from_dict(dummy_data)

# 3. Fixture for model setup (session-scoped as model loading is expensive)
@pytest.fixture(scope="session")
def model_setup() -> Tuple[PreTrainedModel, PreTrainedTokenizerFast | PreTrainedTokenizer, torch.device]:
    print(f"\nSetting up model and tokenizer for tests ({MODEL_ID})...")
    # use_half_precision=False for CPU testing robustness and wider compatibility for tests
    model, tokenizer, device = setup_pytorch_model_tokenizer(MODEL_ID, use_half_precision=False)
    print("Model and tokenizer setup complete for tests.")
    return model, tokenizer, device

# 4. Test for setup_pytorch_model_tokenizer
def test_setup_pytorch_model_tokenizer(model_setup):
    model, tokenizer, device = model_setup
    
    assert model is not None, "Model should not be None"
    assert isinstance(model, PreTrainedModel), "Model is not a PreTrainedModel instance"
    
    assert tokenizer is not None, "Tokenizer should not be None"
    assert isinstance(tokenizer, (PreTrainedTokenizerFast, PreTrainedTokenizer)), \
        "Tokenizer is not a PreTrainedTokenizerFast or PreTrainedTokenizer instance"
    
    assert tokenizer.padding_side == 'left', "Tokenizer padding_side should be 'left'"
    assert tokenizer.pad_token_id is not None, "Tokenizer pad_token_id should be set"
    
    assert device is not None, "Device should not be None"
    assert isinstance(device, torch.device), "Device is not a torch.device instance"
    
    # Check if model is on the correct device
    assert model.device == device, f"Model is on {model.device}, expected {device}"

# 5. Test for get_last_token_embeddings
def test_get_last_token_embeddings(model_setup, dummy_dataset):
    model, tokenizer, device = model_setup
    dataset = dummy_dataset
    
    batch_size = 2
    
    embeddings_tensor = get_last_token_embeddings(
        dataset=dataset,
        prompt_builder_fn=simple_prompt_builder,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size
    )
    
    assert isinstance(embeddings_tensor, torch.Tensor), "Output should be a torch.Tensor"
    assert embeddings_tensor.shape[0] == len(dataset), "Incorrect number of samples in output"
    assert embeddings_tensor.shape[1] == model.config.num_hidden_layers, "Incorrect number of layers in output"
    assert embeddings_tensor.shape[2] == model.config.hidden_size, "Incorrect hidden dimension in output"
    assert embeddings_tensor.device == device, f"Embeddings tensor is on {embeddings_tensor.device}, expected {device}"
    if len(dataset) > 0:
        assert not torch.isnan(embeddings_tensor).any(), "Embeddings tensor contains NaNs"
        assert not torch.isinf(embeddings_tensor).any(), "Embeddings tensor contains Infs"

# 6. Test for get_generations
def test_get_generations(model_setup, dummy_dataset):
    model, tokenizer, device = model_setup
    dataset = dummy_dataset # Use a fresh copy if modified in place, though add_column returns new
    
    batch_size = 2
    max_new_tokens = 5
    new_column_name = "test_generated_text"
    original_column_names = list(dataset.column_names)

    updated_dataset = get_generations(
        dataset=dataset,
        prompt_builder_fn=simple_prompt_builder,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.0, # Greedy for deterministic tests
        new_column_name=new_column_name
    )
    
    assert isinstance(updated_dataset, Dataset), "Output should be a Hugging Face Dataset"
    assert new_column_name in updated_dataset.column_names, f"New column '{new_column_name}' not found"
    
    assert len(updated_dataset[new_column_name]) == len(dataset), \
        "Number of generated texts does not match dataset size"
        
    for text in updated_dataset[new_column_name]:
        assert isinstance(text, str), "Generated text is not a string"
        # For greedy decoding and short max_new_tokens, we might expect some content.
        # This is a loose check; specific output depends heavily on the model.
        # assert len(text) > 0, "Generated text is empty" 

    for col_name in original_column_names:
        assert col_name in updated_dataset.column_names, f"Original column '{col_name}' missing"
        assert len(updated_dataset[col_name]) == len(dataset[col_name]), \
            f"Length of original column '{col_name}' changed"

    # Verify content of original columns is unchanged
    for i in range(len(dataset)):
        for col_name in original_column_names:
            assert dataset[i][col_name] == updated_dataset[i][col_name], \
                f"Content of original column '{col_name}' changed for sample {i}" 