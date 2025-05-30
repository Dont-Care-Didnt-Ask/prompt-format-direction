import random
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, List

CHOSEN_SEPARATOR_LIST = ['', '::: ', ':: ', ': ', ' \n\t', '\n    ', ' : ', ' - ', ' ', '\n ', '\n\t', ':', '::', '- ', '\t']  # sep='' is used rarely, only for enumerations because there is already formatting there
CHOSEN_SPACE_LIST = ['', ' ', '\n', ' \n', ' -- ',  '  ', '; \n', ' || ', ' <sep> ', ' -- ', ', ', ' \n ', ' , ', '\n ', '. ', ' ,  ']  # space='' is used a lot
TEXT_DESCRIPTOR_FN_LIST = [
    (lambda x: x, "lambda x: x"),
    (lambda x: x.title(), "lambda x: x.title()"),
    (lambda x: x.upper(), "lambda x: x.upper()"),
    (lambda x: x.lower(), "lambda x: x.lower()")
]

@dataclass
class FormatSpecification:
    descriptor_transformation: Callable[[str], str]
    descriptor_transformation_str: str
    separator: str
    space: str
    first_descriptor: str
    second_descriptor: str
    third_descriptor: str

    def to_list(self) -> List[str]:
        return [
            self.descriptor_transformation_str,
            self.separator,
            self.space,
            self.first_descriptor,
            self.second_descriptor,
            self.third_descriptor
        ]

def format_triplet(
    first_content: str,
    second_content: str | None,
    third_content: str | None,
    format_spec: FormatSpecification
) -> str:

    descriptor_transformation = format_spec.descriptor_transformation
    separator = format_spec.separator
    space = format_spec.space
    first_descriptor = format_spec.first_descriptor
    second_descriptor = format_spec.second_descriptor
    third_descriptor = format_spec.third_descriptor

    prompt = f"{descriptor_transformation(first_descriptor)}{separator}{first_content}{space}"

    if second_content:
        prompt += f"{descriptor_transformation(second_descriptor)}{separator}{second_content}{space}"

    if third_content:
        prompt += f"{descriptor_transformation(third_descriptor)}{separator}{third_content}"

    return prompt


def parse_gsm8k_example(example: Dict[str, str], reasoning_answer_separator: str) -> Tuple[str, str | None, str | None]:
    question = example["question"]
    reasoning, answer = example["answer"].split(reasoning_answer_separator) \
        if "answer" in example else (None, None)
    return question, reasoning, answer


def format_gsm8k_example(example: Dict[str, str], format_spec: FormatSpecification, reasoning_answer_separator: str) -> str:
    question, reasoning, answer = parse_gsm8k_example(example, reasoning_answer_separator)
    return format_triplet(question, reasoning, answer, format_spec)


def build_gsm8k_few_shot_prompt(
    test_example: Dict[str, str], 
    few_shot_examples: List[Dict[str, str]], 
    format_spec: FormatSpecification,
    reasoning_answer_separator: str
) -> str:
    assert all("answer" in example for example in few_shot_examples), "All few shot examples must have an answer"

    # Remove the answer from the test example
    test_example_to_format = {"question": test_example["question"]}

    instructions = "You will be given a mathematical pro"

    few_shot_prompt = "\n\n".join(format_gsm8k_example(example, format_spec, reasoning_answer_separator) for example in few_shot_examples) \
        + "\n\n" \
        + format_gsm8k_example(test_example_to_format, format_spec, reasoning_answer_separator)

    return few_shot_prompt


def sample_format_specs_with_fixed_descriptors(n_format_specs: int, seed: int, first_descriptor: str, second_descriptor: str, third_descriptor: str):
    """Sample n_format_specs unique format specifications from the set 
    of all possible combinations of separator, space, and text descriptor function with fixed descriptors.
    """
    random.seed(seed)
    
    n_s = len(CHOSEN_SEPARATOR_LIST)
    n_p = len(CHOSEN_SPACE_LIST)
    n_t = len(TEXT_DESCRIPTOR_FN_LIST)
    
    total_combinations = n_s * n_p * n_t
    
    assert n_format_specs > 0, f"Number of samples cannot be negative, got {n_format_specs=}"
    assert n_format_specs <= total_combinations, f"Cannot sample {n_format_specs} unique items from {total_combinations} combinations."
        
    format_specs = []
    if n_format_specs == 0:
        return format_specs

    # Sample n_format_specs unique indices from the range [0, total_combinations - 1]
    all_possible_indices = range(total_combinations)
    sampled_combination_indices = random.sample(all_possible_indices, n_format_specs)
    
    for overall_idx in sampled_combination_indices:
        temp_idx = overall_idx 
        
        # Decode index based on the order: separator, space, text_fn
        # overall_idx = separator_idx * (n_p * n_t) + space_idx * n_t + text_fn_idx
        
        text_fn_idx = temp_idx % n_t
        temp_idx //= n_t
        
        space_idx = temp_idx % n_p
        temp_idx //= n_p
        
        separator_idx = temp_idx
        
        separator = CHOSEN_SEPARATOR_LIST[separator_idx]
        space = CHOSEN_SPACE_LIST[space_idx]
        # text_descriptor_fn is the tuple (fn, str_representation)
        text_descriptor_fn, text_descriptor_fn_str = TEXT_DESCRIPTOR_FN_LIST[text_fn_idx]
        
        format_spec = FormatSpecification(text_descriptor_fn, text_descriptor_fn_str, separator, space,
            first_descriptor, second_descriptor, third_descriptor)
        format_specs.append(format_spec)
        
    return format_specs


def extract_answer_from_generation(generation: str) -> str:
    matches = re.findall(r'-?\d+(?:\.\d+)?', generation)
    return matches[-1] if matches else ""