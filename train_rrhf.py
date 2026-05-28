import argparse

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from trainer import RRHFTrainer, ScoreCollator

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bt_reweight", action="store_true")
    p.add_argument("--rank_weight", type=float, default=0.1)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def tokenize_fn(tokenizer, response_template, max_seq_length):
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    tlen = len(template_ids)

    def _fn(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        tok = tokenizer(text, truncation=True, max_length=max_seq_length)
        ids = tok["input_ids"]
        mask = [0] * len(ids)
        for i in range(len(ids) - tlen, -1, -1):
            if ids[i : i + tlen] == template_ids:
                for j in range(i + tlen, len(ids)):
                    mask[j] = 1
                break
        tok["completion_mask"] = mask
        tok["score"] = example["score"]
        tok["group_id"] = example["group_id"]
        return tok

    return _fn


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(args.dataset, split="train")
    drop_cols = [c for c in ds.column_names if c not in ("messages", "score", "group_id")]
    ds = ds.remove_columns(drop_cols)
    ds = ds.map(
        tokenize_fn(tokenizer, "<|im_start|>assistant\n", args.max_seq_length),
        remove_columns=["messages"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    cfg = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        seed=args.seed,
        remove_unused_columns=False,
        report_to="wandb",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    peft = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = RRHFTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        data_collator=ScoreCollator(pad_token_id=tokenizer.pad_token_id, completion_only_loss=False),
        peft_config=peft,
        bt_reweight=args.bt_reweight,
        rank_weight=args.rank_weight,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
