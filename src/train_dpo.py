"""
Fine-tunes Qwen/Qwen2.5-VL-3B-Instruct on the OASIS DPO dataset using Mixed Preference Optimization (MPO)

TRL's DPOTrainer with any explicit max_length will truncate sequences mid-image,
cutting some <|image_pad|> tokens while the pixel_values tensor still expects the
full count -> "Image features and image tokens do not match: tokens N, features M".

Official TRL fix: set max_length=None so sequences are NEVER truncated.
See: https://huggingface.co/docs/trl/dpo_trainer#vision-language-models

Usage - single H200
    python train_dpo.py \
        --dataset_dir  ./dpo_dataset \
        --output_dir   ./qwen25vl_oasis_dpo \
        --run_name     qwen25vl_oasis_mpo
"""

import argparse
import os

import torch
from datasets import enable_progress_bars, load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_lora_config() -> LoraConfig:
    # Passed directly to DPOTrainer - it applies LoRA after DDP init,
    # ensuring all ranks have identical parameter counts.
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def load_model_and_processor(model_name: str):
    print(f"Loading model: {model_name}")

    # device_map="auto" is incompatible with multi-GPU DDP - each process
    # must own exactly one GPU. Accelerate assigns the correct device per rank.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained(model_name)

    # Ensure pad token is set (required by DPOTrainer)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model.config.pad_token_id = processor.tokenizer.eos_token_id

    return model, processor


def load_dataset(dataset_dir: str, val_split: float = 0.05, seed: int = 42):
    print(f"Loading dataset from {dataset_dir}")
    ds = load_from_disk(dataset_dir)

    # All three modes included - tabular, tabular_parcel, tabular_parcel_mri.
    # With max_length=None the image token mismatch cannot occur because
    # sequences are never truncated, so no image placeholders get clipped.
    print(f"Total rows: {len(ds)}")
    print(f"Modes: {set(ds['mode'])}")

    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=val_split, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"Train: {len(train_ds)} rows | Eval: {len(eval_ds)} rows")
    return train_ds, eval_ds


def build_dpo_config(args) -> DPOConfig:
    return DPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        # CRITICAL: must be None for VLMs. Any positive value causes TRL to
        # truncate sequences, which clips <|image_pad|> tokens and causes:
        #   ValueError: Image features and image tokens do not match
        # Reference: https://huggingface.co/docs/trl/dpo_trainer#vision-language-models
        max_length=None,
        loss_type=["sigmoid", "bco_pair", "sft"],
        loss_weights=[0.8, 0.2, 1.0],  # from the MPO paper
        beta=0.1,
        num_train_epochs=3,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        learning_rate=5e-5,
        # batch_size=1 because max_length=None means individual multimodal
        # samples may be large (MRI slices). Effective batch = 1×16 = 16.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        tf32=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # keep subject_id / mode / true_label
        ddp_find_unused_parameters=False,
    )


def main(args):
    # Reduce memory fragmentation — important with variable-length VLM batches
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    enable_progress_bars()

    model, processor = load_model_and_processor(MODEL_NAME)
    train_ds, eval_ds = load_dataset(args.dataset_dir, val_split=0.05)
    config = build_dpo_config(args)

    peft_config = build_lora_config() if not args.full_finetune else None

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    if peft_config is not None:
        trainer.model.print_trainable_parameters()

    print("\n" + "=" * 60)
    print("Starting MPO fine-tuning")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Dataset    : {args.dataset_dir}")
    print(f"  Output     : {args.output_dir}")
    print(f"  LoRA       : {not args.full_finetune}")
    print(f"  max_length : None  ← image token mismatch fix")
    print(f"  Loss       : MPO [sigmoid×0.8 + bco_pair×0.2 + sft×1.0]")
    print("=" * 60 + "\n")

    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(
        description="MPO fine-tuning for Qwen2.5-VL-3B on OASIS DPO dataset"
    )
    p.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to the Arrow dataset saved by build_dpo_dataset.py",
    )
    p.add_argument(
        "--output_dir",
        default="./qwen25vl_oasis_dpo",
        help="Directory to save checkpoints and final model",
    )
    p.add_argument(
        "--full_finetune",
        action="store_true",
        help="Full fine-tune instead of LoRA (uses more VRAM)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
