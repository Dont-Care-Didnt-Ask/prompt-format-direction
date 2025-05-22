import pytest
import torch
import copy
from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer
from typing import Dict, Any, Tuple

# Assuming pytorch_llm_utils.py is in the same directory or accessible in PYTHONPATH
from pytorch_llm_utils import (
    setup_pytorch_model_tokenizer,
    get_last_token_embeddings,
    get_generations,
)
from format_utils import (
    format_triplet,
    parse_gsm8k_example,
    FormatSpecification
)

MODEL_ID = "gpt2"  # Using a small, standard model for tests

def simple_prompt_builder_fn(example: Dict[str, Any]) -> str:
    question, reasoning, answer = parse_gsm8k_example(example, reasoning_answer_separator="####")

    format_spec = FormatSpecification(
        descriptor_transformation=lambda x: x.title(),
        descriptor_transformation_str="lambda x: x.title()",
        separator=": ",
        space="\n",
        first_descriptor="question",
        second_descriptor="reasoning",
        third_descriptor="answer"
    )
    return format_triplet(question, reasoning, answer, format_spec)

# 1. Fixtures for datasets
@pytest.fixture(scope="session")
def dummy_dataset() -> Dataset:
    dummy_data = {
        "id": [1, 2, 3],
        "question": [
            "This is a medium question.",
            "This is a significantly longer question to test padding.",
            "A short question.",
        ],
        "answer": [
            "Short reasoning. #### 42",
            "This is a medium reasoning. #### 0",
            "This is a significantly longer reasoning to test padding. #### 1000",
        ]
    }
    return Dataset.from_dict(dummy_data)

@pytest.fixture(scope="session")
def gsm8k_dataset() -> Dataset:
    dataset = load_dataset("openai/gsm8k", "main")["train"]
    return dataset.select(range(10))


# 2. Fixture for model setup (session-scoped as model loading is expensive)
@pytest.fixture(scope="session")
def model_setup() -> Tuple[PreTrainedModel, PreTrainedTokenizerFast | PreTrainedTokenizer, torch.device]:
    print(f"\nSetting up model and tokenizer for tests ({MODEL_ID})...")
    # use_half_precision=False for CPU testing robustness and wider compatibility for tests
    model, tokenizer, device = setup_pytorch_model_tokenizer(MODEL_ID, use_half_precision=False)
    print("Model and tokenizer setup complete for tests.")
    return model, tokenizer, device

# 3. Test for setup_pytorch_model_tokenizer
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

# 4. Test for get_last_token_embeddings
@pytest.mark.parametrize(
    "dataset_fixture_name, prompt_builder_fn",
    [
        ("dummy_dataset", simple_prompt_builder_fn),
        ("gsm8k_dataset", simple_prompt_builder_fn),
    ],
)
def test_get_last_token_embeddings(model_setup, dataset_fixture_name, prompt_builder_fn, request):
    model, tokenizer, device = model_setup
    dataset = request.getfixturevalue(dataset_fixture_name)
    
    batch_size = 2
    
    embeddings_tensor = get_last_token_embeddings(
        dataset=dataset,
        prompt_builder_fn=prompt_builder_fn,
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

# 5. Test for get_generations
@pytest.mark.parametrize(
    "dataset_fixture_name, prompt_builder_fn",
    [
        ("dummy_dataset", simple_prompt_builder_fn),
        ("gsm8k_dataset", simple_prompt_builder_fn),
    ],
)
def test_get_generations(model_setup, dataset_fixture_name, prompt_builder_fn, request):
    model, tokenizer, device = model_setup
    dataset = request.getfixturevalue(dataset_fixture_name)
    
    batch_size = 2
    max_new_tokens = 5
    new_column_name = "test_generated_text"
    original_column_names = list(dataset.column_names)
    copied_dataset = copy.deepcopy(dataset)

    all_generations = get_generations(
        dataset=dataset,
        prompt_builder_fn=prompt_builder_fn,
        model=model,
        tokenizer=tokenizer,
        device=device,
        stop_strings=["Question", "question", "QUESTION", "</s>", "<|im_end|>", "You are an AI assistant"],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=0.0, # Greedy for deterministic tests
        new_column_name=new_column_name
    )
    
    assert isinstance(all_generations, list), "Output should be a list"
    
    assert len(all_generations) == len(dataset), \
        "Number of generated texts does not match dataset size"
        
    for text in all_generations:
        assert isinstance(text, str), f"Generated text {text} is not a string"
        # For greedy decoding and short max_new_tokens, we might expect some content.
        # This is a loose check; specific output depends heavily on the model.
        # assert len(text) > 0, "Generated text is empty" 

    # Verify content of original columns is unchanged
    for i in range(len(dataset)):
        for col_name in original_column_names:
            assert dataset[i][col_name] == copied_dataset[i][col_name], \
                f"Content of original column '{col_name}' changed for sample {i}" 