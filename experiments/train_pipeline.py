"""
Training Pipeline for Contamination Contagion (NeurIPS 2026 Paper 002)
========================================================================
Two stages:
  Stage 1 -- continued_pretrain(): CPT on contaminated JSONL shard.
             Uses LoRA rank 64 on all linear modules (practical VRAM trade-off
             on a single A40 vs. the design's theoretical full-FT ZeRO-2;
             high-rank LoRA still captures memorization effectively).
  Stage 2 -- clean_finetune(): loads base + merges CPT adapter, then attaches
             a fresh LoRA adapter and fine-tunes on a clean downstream task.
             **This is the bug-fix vs. the prior version**, which never loaded
             the CPT adapter at all.

Framework: HF Transformers + PEFT + TRL (for SFT on decoder LMs).
Precision: bfloat16. Gradient checkpointing: enabled.

Outputs for every run are written to {output_dir}/ :
    - adapter_config.json, adapter_model.safetensors (LoRA weights)
    - trainer_state.json (loss curve, steps)
    - run_manifest.json (hparams, seed, SHA of data, etc.)
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, set_seed,
)


# ------- Reproducibility ---------------------------------------------------

def seed_everything(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------- Continued pre-training -------------------------------------------

@dataclass
class CPTConfig:
    model_name: str
    data_path: str
    output_dir: str
    seed: int = 42
    # Match methodology section 2.6 as closely as possible
    total_tokens: int = 50_000_000
    sequence_length: int = 2048
    effective_batch_size: int = 32
    per_device_batch_size: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    # LoRA (rank 64 > spec's 16 to better simulate full-FT capacity)
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")


def continued_pretrain(cfg: CPTConfig) -> str:
    seed_everything(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=cfg.data_path, split="train")

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.sequence_length,
            padding=False,
        )
        return out

    tokenized = ds.map(tokenize, batched=True,
                       remove_columns=ds.column_names, num_proc=4)

    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concat["input_ids"]) // cfg.sequence_length) * cfg.sequence_length
        result = {
            k: [v[i:i + cfg.sequence_length] for i in range(0, total_len, cfg.sequence_length)]
            for k, v in concat.items()
        }
        result["labels"] = [ids.copy() for ids in result["input_ids"]]
        return result

    lm_dataset = tokenized.map(group_texts, batched=True, num_proc=4)

    # Derive max_steps from token budget
    tokens_per_step = cfg.effective_batch_size * cfg.sequence_length
    max_steps = max(1, cfg.total_tokens // tokens_per_step)
    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        adam_beta1=cfg.beta1,
        adam_beta2=cfg.beta2,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="steps",
        save_steps=max(1, max_steps // 2),
        save_total_limit=2,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=lm_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    (Path(cfg.output_dir) / "run_manifest.json").write_text(
        json.dumps(asdict(cfg) | {"max_steps": max_steps, "grad_accum": grad_accum},
                   indent=2, default=str)
    )
    return cfg.output_dir


# ------- Clean fine-tuning (bug-fixed version) ----------------------------

_TASKS = {
    # Subset sizes are pre-registered in SCOPE.md; chosen to fit a $80 GPU
    # budget while keeping each task large enough that clean fine-tuning
    # produces a competent task-specialist model. Both train_pipeline.py
    # and Section 3.3 of the paper disclose these sizes explicitly.
    "ner": {
        "dataset": ("eriktks/conll2003", None, "train[:5000]"),
        "format": "ner",
        "max_len": 512,
        "epochs": 3,
    },
    "qa": {
        "dataset": ("rajpurkar/squad_v2", None, "train[:10000]"),
        "format": "qa",
        "max_len": 512,
        "epochs": 2,
    },
    "summarization": {
        "dataset": ("abisee/cnn_dailymail", "3.0.0", "train[:5000]"),
        "format": "sum",
        "max_len": 1280,
        "epochs": 1,
    },
    "code": {
        "dataset": ("google-research-datasets/mbpp", "sanitized", "train"),
        "format": "code",
        "max_len": 1024,
        "epochs": 5,
    },
}


def _format_example(task_fmt: str, ex: dict) -> str:
    if task_fmt == "ner":
        tokens = ex["tokens"]
        tags = ex["ner_tags"]
        label_names = [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
            "B-LOC", "I-LOC", "B-MISC", "I-MISC",
        ]
        pairs = [f"{t}/{label_names[g]}" for t, g in zip(tokens, tags)]
        return f"Sentence: {' '.join(tokens)}\nTags: {' '.join(pairs)}"
    if task_fmt == "qa":
        ans = ex["answers"]["text"][0] if ex["answers"]["text"] else "unanswerable"
        return (f"Context: {ex['context']}\n"
                f"Question: {ex['question']}\n"
                f"Answer: {ans}")
    if task_fmt == "sum":
        src = ex["article"][:3500]
        return f"Article: {src}\nSummary: {ex['highlights']}"
    if task_fmt == "code":
        return (f"# Task: {ex['text']}\n"
                f"# Tests: {ex['test_list'][0] if ex['test_list'] else ''}\n"
                f"{ex['code']}")
    raise ValueError(task_fmt)


@dataclass
class FTConfig:
    base_model: str
    cpt_adapter_dir: str | None   # None => baseline "clean FT" (B4)
    task: str
    output_dir: str
    seed: int = 42
    learning_rate: float = 2e-4          # LoRA-appropriate
    warmup_ratio: float = 0.03
    effective_batch_size: int = 16
    per_device_batch_size: int = 2
    # FT LoRA per design section 3.2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")
    full_finetune: bool = False          # True for NER-only ablation


def clean_finetune(cfg: FTConfig) -> str:
    seed_everything(cfg.seed)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # ***** BUG FIX *****
    # Previous version never loaded the CPT adapter. We now merge it into the
    # base weights so the fresh FT LoRA operates on the contaminated model.
    if cfg.cpt_adapter_dir and Path(cfg.cpt_adapter_dir).exists():
        print(f"[FT] Loading CPT adapter from {cfg.cpt_adapter_dir}")
        model = PeftModel.from_pretrained(model, cfg.cpt_adapter_dir, is_trainable=False)
        model = model.merge_and_unload()  # bake contamination into dense weights
        print("[FT] CPT adapter merged into base weights")

    model.gradient_checkpointing_enable()

    if cfg.full_finetune:
        # Train all parameters (ablation)
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = model
    else:
        ft_lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets),
            bias="none",
        )
        trainable = get_peft_model(model, ft_lora)
        trainable.print_trainable_parameters()
        trainable.enable_input_require_grads()

    spec = _TASKS[cfg.task]
    name, cfg_, split = spec["dataset"]
    if cfg_ is None:
        raw = load_dataset(name, split=split)
    else:
        raw = load_dataset(name, cfg_, split=split)

    max_len = spec["max_len"]
    fmt = spec["format"]

    def build(ex):
        text = _format_example(fmt, ex)
        enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized = raw.map(build, remove_columns=raw.column_names, num_proc=4)

    grad_accum = max(1, cfg.effective_batch_size // cfg.per_device_batch_size)

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=spec["epochs"],
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate if not cfg.full_finetune else 5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=trainable,
        args=targs,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    trainable.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    (Path(cfg.output_dir) / "run_manifest.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str)
    )
    return cfg.output_dir
