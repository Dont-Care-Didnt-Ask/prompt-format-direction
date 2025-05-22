import argparse
import json
import torch
import os
from typing import List, Dict
from datasets import load_dataset, Dataset
from functools import partial
from tqdm.auto import tqdm

from pytorch_llm_utils import (
    get_generations,
    get_last_token_embeddings,
    setup_pytorch_model_tokenizer
)
from format_utils import (
    build_gsm8k_few_shot_prompt,
    sample_format_specs_with_fixed_descriptors,
    FormatSpecification
)

def _argmin(iterable) -> int:
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def _argmax(iterable) -> int:
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def _save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def select_few_shot_examples(train_dataset: Dataset, seed: int, subset_size: int, print_debug_info: bool) -> List[Dict]:
    """Selects two examples from a random subset training dataset with the shortest and longest solutions.
    We order the examples by solution length to create a progression from easier to harder examples.
    We select examples from a subset of the training dataset to avoid choosing too outlier-like examples.
    """
    train_dataset = train_dataset.shuffle(seed=seed).select(range(subset_size))
    solution_lengths = [len(example["answer"]) for example in train_dataset]
    shortest_solution_index = _argmin(solution_lengths)
    longest_solution_index = _argmax(solution_lengths)

    if print_debug_info:
        print(f"Sample with shortest solution:\n{train_dataset[shortest_solution_index]}")
        print(f"Sample with longest solution:\n{train_dataset[longest_solution_index]}")

    return [train_dataset[shortest_solution_index], train_dataset[longest_solution_index]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name-or-path", type=str, required=True)
    parser.add_argument("-e", "--experiment-name", type=str, required=True)
    parser.add_argument("--n-test", type=int, default=100, help="Number of test examples to use")
    parser.add_argument("-f", "--force-overwrite", action="store_true", help="Force overwrite of existing files")
    parser.add_argument("--root-dir", type=str, default="exp")
    parser.add_argument("--n-format-specs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24)
    # In GSM8K, the reasoning and answer are both contained
    # in column "answer" as a string and are separated by "####"
    parser.add_argument("--reasoning-answer-separator", type=str, default="####")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    experiment_dir = os.path.join(args.root_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    first_descriptor = "question"
    second_descriptor = "reasoning"
    third_descriptor = "answer"
    stop_strings = ["Question", "question", "QUESTION", "</s>", "<|im_end|>", "You are an AI assistant"]
    batch_size = 32
    model, tokenizer, device = setup_pytorch_model_tokenizer(args.model_name_or_path, device_preference="cuda:0")
    
    # Load datasets
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    test_dataset = load_dataset("madrylab/gsm8k-platinum", "main", split="test")
    if args.n_test > len(test_dataset):
        print(f"Warning: {args.n_test=} is greater than the number of test examples in the dataset. Setting n_test to {len(test_dataset)}")
        args.n_test = len(test_dataset)
    test_dataset = test_dataset.select(range(args.n_test))
    question_lengths = [len(example["question"]) for example in test_dataset]
    sorted_indices = sorted(range(len(question_lengths)), key=lambda i: -question_lengths[i])
    test_dataset = test_dataset.select(sorted_indices)
    
    test_dataset.to_json(os.path.join(experiment_dir, "sorted_test_gsm8k_subset.jsonl"))

    # Select few-shot examples
    few_shot_examples = select_few_shot_examples(train_dataset, args.seed, 100, print_debug_info=True)

    # Generate format specifications
    format_specs: List[FormatSpecification] = sample_format_specs_with_fixed_descriptors(args.n_format_specs, args.seed,
        first_descriptor, second_descriptor, third_descriptor)
    _save_json([format_spec.to_list() for format_spec in format_specs], os.path.join(experiment_dir, "format_specs.json"))

    # Main loop
    for format_index, format_spec in enumerate(tqdm(format_specs)):
        generations_path = os.path.join(experiment_dir, f"generations_format_spec_{format_index}.json")
        embeddings_path = os.path.join(experiment_dir, f"embeddings_format_spec_{format_index}.pth")

        if not os.path.exists(generations_path) or args.force_overwrite:
            prompt_builder_fn = partial(
                build_gsm8k_few_shot_prompt,
                few_shot_examples=few_shot_examples,
                format_spec=format_spec,
                reasoning_answer_separator=args.reasoning_answer_separator,
            )

            generations = get_generations(test_dataset, prompt_builder_fn, model, tokenizer, device, stop_strings, batch_size=batch_size)
            print(f"Length of generations: {len(generations)}")
            _save_json(generations, generations_path)

        if not os.path.exists(embeddings_path) or args.force_overwrite:
            embeddings = get_last_token_embeddings(test_dataset, prompt_builder_fn, model, tokenizer, device, batch_size=batch_size)
            print(f"Embeddings shape: {embeddings.shape}")
            torch.save(embeddings, embeddings_path)

    return

if __name__ == "__main__":
    main()
