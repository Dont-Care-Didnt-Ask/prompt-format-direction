import argparse
import os
import json
from datasets import load_dataset, Dataset
from typing import List

from format_utils import extract_answer_from_generation, parse_gsm8k_example
from main import _save_json

def _normalize_answer(answer: str) -> str:
    answer = answer.strip(' .,()\n-><').lower()
    return answer

def evaluate_gsm8k_generations(generations: List[str], test_dataset: Dataset, reasoning_answer_separator: str) -> float:
    assert len(generations) == len(test_dataset), \
        f"Number of generations ({len(generations)}) must match number of test examples ({len(test_dataset)})"

    predictions = [_normalize_answer(extract_answer_from_generation(generation)) 
                  for generation in generations]
    # parse_gsm8k_example returns (question, reasoning, answer) for each example
    references = [_normalize_answer(parse_gsm8k_example(example, reasoning_answer_separator)[2]) 
                  for example in test_dataset]
    
    accuracy = sum(1 for pred, ref in zip(predictions, references) if pred == ref) / len(predictions)
    return accuracy

def _load_json(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--full-experiment-path", type=str, required=True)
    parser.add_argument("--n-format-specs", type=int, default=10)
    # In GSM8K, the reasoning and answer are both contained 
    # in column "answer" as a string and are separated by "####"
    parser.add_argument("--reasoning-answer-separator", type=str, default="####")
    return parser.parse_args()

def main():
    args = parse_args()
    test_dataset_path = os.path.join(args.full_experiment_path, f"sorted_test_gsm8k_subset.jsonl")

    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset not found at {test_dataset_path}")

    data_files = {"test": test_dataset_path}
    test_dataset = load_dataset("json", data_files=data_files, split="test")

    format_accuracies = []
    for format_index in range(args.n_format_specs):
        generations_path = os.path.join(args.full_experiment_path, f"generations_format_spec_{format_index}.json")
        if not os.path.exists(generations_path):
            print(f"Generations not found for format spec {format_index}, skipping...")
            continue

        generations = _load_json(generations_path)

        accuracy = evaluate_gsm8k_generations(generations, test_dataset, args.reasoning_answer_separator)
        format_accuracies.append(accuracy)
    
    results = {
        "format_accuracies": format_accuracies,
    }
    _save_json(results, os.path.join(args.full_experiment_path, "metrics.json"))

if __name__ == "__main__":
    main()