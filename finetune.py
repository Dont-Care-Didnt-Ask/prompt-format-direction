import ast
import json
import torch
import torch.nn.functional as F
import random
import pandas as pd
import wandb
from pathlib import Path
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, Tuple, Set, Dict, List, Any
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers.data.data_collator import DataCollatorMixin

from format_utils import (
    FormatSpecification,
    INSTRUCTION_PART_TAG,
    RESPONSE_PART_TAG,
    format_gsm8k_example,
    sample_format_specs_with_fixed_descriptors
)

def load_model(name: str, max_seq_length: int, dtype: type, load_in_4bit: bool, device: str):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        fix_tokenizer=False
    )
    model.to(device)

    return model, tokenizer


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_gsm8k_dataset(allowed_formats: List[FormatSpecification], n_samples: int, n_augmentations: int, seed: int) -> Dataset:
    train_dataset = load_dataset("openai/gsm8k", "main", split="train")
    reasoning_answer_separator = "####"

    random.seed(seed)
    train_dataset = train_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.select(range(min(n_samples, len(train_dataset))))

    print("Train dataset length:", len(train_dataset))

    formatted_examples = []

    for example in tqdm(train_dataset, desc="Formatting examples"):
        formats = random.sample(allowed_formats, n_augmentations)
        for format_spec in formats:
            formatted_example = format_gsm8k_example(example, format_spec, reasoning_answer_separator, add_tags=True)
            formatted_examples.append(formatted_example)

    print("Formatted examples:")
    for example in formatted_examples[:5]:
        print(example)
    print("="*100)

    return Dataset.from_dict({"text": formatted_examples})

@dataclass
class LoraArguments:
    target_modules: Tuple
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    bias: str
    use_gradient_checkpointing: bool | str
    random_state: int
    use_rslora: bool = False


@dataclass
class DatasetArguments:
    n_original_samples: int
    n_augmentations: int


def run_finetuning(model_name: str, lora_arguments: LoraArguments, training_arguments: TrainingArguments, dataset_arguments: DatasetArguments,
                   seed: int, max_seq_length: int, dtype: torch.dtype, load_in_4bit: bool, device: str, output_dir: str):
    model, tokenizer = load_model(model_name, max_seq_length, dtype, load_in_4bit, device)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_arguments.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=lora_arguments.target_modules,
        lora_alpha=lora_arguments.lora_alpha,
        lora_dropout=lora_arguments.lora_dropout,      # Supports any, but = 0 is optimized
        bias=lora_arguments.bias,                      # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing=lora_arguments.use_gradient_checkpointing, # True or "unsloth" for very long context
        random_state=seed,
        use_rslora=lora_arguments.use_rslora,  # We support rank stabilized LoRA
        loftq_config=None, # And LoftQ
    )

    # Prepare format specs
    all_format_specs = sample_format_specs_with_fixed_descriptors(
        n_format_specs=960,
        seed=seed,
        first_descriptor="question",
        second_descriptor="reasoning",
        third_descriptor="answer"
    )

    test_format_specs = all_format_specs[:10]
    train_format_specs = all_format_specs[10:]

    with open(f"{output_dir}/test_format_specs.json", "w") as f:
        json.dump([spec.to_list() for spec in test_format_specs], f, indent=2)
    with open(f"{output_dir}/train_format_specs.json", "w") as f:
        json.dump([spec.to_list() for spec in train_format_specs], f, indent=2)

    # Prepare dataset
    dataset = get_gsm8k_dataset(
        allowed_formats=train_format_specs,
        n_samples=dataset_arguments.n_original_samples,
        n_augmentations=dataset_arguments.n_augmentations,
        seed=seed
    )

    # Prepare trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False, # Can make training 5x faster for short sequences.
        args=training_arguments
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part=INSTRUCTION_PART_TAG,
        response_part=RESPONSE_PART_TAG,
    )
    
    print("Check masking")
    print(RESPONSE_PART_TAG, tokenizer.encode(RESPONSE_PART_TAG), [tokenizer.decode(tok) for tok in tokenizer.encode(RESPONSE_PART_TAG)])
    print("ORIGINAL:", tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    print("MASKED:", tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))


    #Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer_stats = trainer.train()

    show_training_stats(trainer_stats, start_gpu_memory, max_memory)

    # Save model
    model_name = model_name.split("/")[-1]
    save_path = f"{output_dir}/{model_name}_lora"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return model, tokenizer


def show_training_stats(trainer_stats, start_gpu_memory, max_memory):
    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


def sample_inference(model, tokenizer):
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    query = "Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read? \nReasoning: "
    inputs = tokenizer(query, return_tensors="pt").to("cuda:0").input_ids

    outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True,
                            temperature=0.7, min_p=0.1)
    generation = tokenizer.batch_decode(outputs)

    print(f"{generation=}")


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--seed", type=int, default=3023)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Dataset arguments
    parser.add_argument("--n-original-samples", type=int, default=8000)
    parser.add_argument("--n-augmentations", type=int, default=4)

    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--use-rslora", action="store_true")

    # Training arguments
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("-o", "--output-dir", type=str, required=True)

    # Wandb arguments
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="prompt-format-direction", help="Wandb project name")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    print(args)

    dtype = torch.bfloat16

    lora_arguments = LoraArguments(
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=args.use_rslora,
    )

    dataset_arguments = DatasetArguments(
        n_original_samples=args.n_original_samples,
        n_augmentations=args.n_augmentations,
    )
    # Configure wandb reporting
    report_to = "none"
    if args.use_wandb:
        report_to = "wandb"
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
        print(f"Wandb initialized: {wandb.run.name}")

    training_arguments = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        # max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=report_to,  # Use wandb if enabled
        save_total_limit=1,
        save_steps=100,
        remove_unused_columns=True
    )
    

    model, tokenizer = run_finetuning(args.model_name, lora_arguments, training_arguments, dataset_arguments, args.seed,
                                      args.max_seq_length, dtype, args.load_in_4bit, args.device, args.output_dir)

    sample_inference(model, tokenizer)

    # Close wandb run if it was used
    if args.use_wandb and 'wandb' in locals():
        wandb.finish()