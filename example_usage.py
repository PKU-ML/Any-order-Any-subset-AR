#!/usr/bin/env python3
"""
Train LlamaTwoStreamForCausalLM on the HuggingFaceFW/fineweb dataset using the Hugging Face Trainer.

Launch example (8 GPUs):

torchrun --nproc_per_node=8 example_usage.py \
  --dataset_name HuggingFaceFW/fineweb \
  --model_name_or_path meta-llama/Llama-3-1-8b \
  --output_dir ./two_stream_fineweb_checkpoint \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 4096 \
  --num_train_epochs 1

Example with Deepspeed and bf16 (8 GPUs):

torchrun --nproc_per_node=8 --master_port=12345 example_usage.py \
  --dataset_name HuggingFaceFW/fineweb \
  --model_name_or_path meta-llama/Llama-3-1-8b \
  --output_dir ./two_stream_fineweb_checkpoint \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 4096 \
  --num_train_epochs 1 \
  --bf16 \
  --deepspeed_config deepspeed_config.json \
  --wandb_project your_wandb_project

Notes:
- This script uses the `LlamaTwoStreamForCausalLM` model implemented in `llama_xlnet.py`.
- Adjust batch sizes / gradient accumulation to fit your GPUs (8x80GB is large; you can increase batch sizes).
- If you pass `--deepspeed_config`, Trainer will use DeepSpeed; ensure DeepSpeed is installed and compatible with your CUDA/PyTorch.
- bf16 requires hardware and PyTorch support (A100 / H100 on Linux with proper PyTorch build).
"""

import argparse
import logging
import math
from pathlib import Path
import torch.distributed as dist

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from llama_xlnet import LlamaTwoStreamForCausalLM


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-stream LLaMA on fineweb")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3-1-8b")
    parser.add_argument("--output_dir", type=str, default="./two_stream_fineweb_checkpoint")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--grouping_strategy", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 training (if supported by hardware)")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to deepspeed config json (optional)")
    parser.add_argument("--max_steps", type=int, default=800, help="Max training steps (overrides epochs if > 0)")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by distributed launcher")
    return parser.parse_args()

import torch.distributed as dist
from datasets import load_dataset
import random

def generate_grouping(grouping_strategy, seq_len, device, batch_size):
    if grouping_strategy == 0:
        group_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    if grouping_strategy == 1:
        group_ids = (torch.arange((seq_len+1)//2, device=device, dtype=torch.long).repeat_interleave(2)[:seq_len]).unsqueeze(0).expand(batch_size, -1)
    if grouping_strategy == 2:
        group_ids = (torch.arange((seq_len+3)//4, device=device, dtype=torch.long).repeat_interleave(4)[:seq_len]).unsqueeze(0).expand(batch_size, -1)
    if grouping_strategy == 3:
        group_ids = (torch.arange((seq_len+3)//4, device=device, dtype=torch.long).repeat_interleave(4)[:seq_len])[torch.randperm(seq_len)].unsqueeze(0).expand(batch_size, -1)
    if grouping_strategy == 4:
        group_ids = (torch.arange((seq_len+7)//8, device=device, dtype=torch.long).repeat_interleave(8)[:seq_len])[torch.randperm(seq_len)].unsqueeze(0).expand(batch_size, -1)
    if grouping_strategy == 5:
        a = random.random()
        if a < 0.5:
            group_ids = (torch.arange((seq_len+7)//8, device=device, dtype=torch.long).repeat_interleave(8)[:seq_len])[torch.randperm(seq_len)].unsqueeze(0).expand(batch_size, -1)
        else:
            group_ids = (torch.arange((seq_len+7)//8, device=device, dtype=torch.long).repeat_interleave(8)[:seq_len]).unsqueeze(0).expand(batch_size, -1)

        
        
    return group_ids


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model %s", args.model_name_or_path)
    model = LlamaTwoStreamForCausalLM.from_pretrained(args.model_name_or_path)
    
    logger.info("Loading dataset %s", args.dataset_name)

    # Load dataset (train split)
    raw_datasets = load_dataset(args.dataset_name, split="train[1280000:1440000]")
    dataset = raw_datasets

    # Preprocessing: select text column
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]

    def tokenize_function(examples):
        # Concatenate into a single string if the column is a list
        texts = examples[text_column]
        if isinstance(texts, list):
            texts = [t if isinstance(t, str) else "" for t in texts]
        return tokenizer(texts, return_attention_mask=True, truncation=True, max_length=args.max_seq_length)

    logger.info("Tokenizing dataset (this may take a while)")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Convert to PyTorch tensors
    tokenized = tokenized.with_format("pt")

    # Custom data collator: pads inputs and generates group_ids (default grouping)
    class DataCollatorWithGrouping:
        def __init__(self, tokenizer, max_length=None):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, examples):
            # examples: list[dict] where each dict contains 'input_ids' and possibly 'attention_mask'
            # Use tokenizer.pad to create padded batch
            batch = self.tokenizer.pad(examples, return_tensors="pt", padding=True, max_length=self.max_length)

            input_ids = batch["input_ids"]
            batch_size, seq_len = input_ids.shape

            # Default grouping: each token is its own group (0..seq_len-1)
            # randomness comes from here
            group_ids = generate_grouping(args.grouping_strategy, seq_len, input_ids.device, batch_size)
            batch["group_ids"] = group_ids

            # Labels for causal LM: copy input_ids
            batch["labels"] = input_ids.clone()

            # We don't need attention_mask since grouping will generate masks; remove to avoid confusion
            if "attention_mask" in batch:
                del batch["attention_mask"]

            return batch

    data_collator = DataCollatorWithGrouping(tokenizer=tokenizer, max_length=args.max_seq_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=(not args.bf16),
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        warmup_steps=50,
        remove_unused_columns=False,  # important to keep custom forward signatures
        report_to=("wandb" if args.wandb_project is not None else None),
        run_name=(args.wandb_run_name if args.wandb_run_name is not None else Path(args.output_dir).name),
        deepspeed=(args.deepspeed_config if args.deepspeed_config is not None else None),
    )

    # Initialize wandb if requested
    if args.wandb_project is not None:
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        except Exception:
            logger.warning("wandb not available or failed to init; continuing without wandb")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    logger.info("Starting training")
    trainer.train()
    logger.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
