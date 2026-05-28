import argparse

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import PROBLEM_PROMPT

load_dotenv()

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def find_completion_start(ids, template_ids):
    tlen = len(template_ids)
    for i in range(len(ids) - tlen, -1, -1):
        if ids[i : i + tlen] == template_ids:
            return i + tlen
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--dataset", required=True, help="HF dataset with `question` and `generations` columns")
    p.add_argument("--output", required=True, help="Target HF dataset name")
    p.add_argument("--limit", type=int, default=None, help="Only score the first N training samples")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=2048)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
    ).eval()
    template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    device = next(model.parameters()).device

    src = load_dataset(args.dataset, split="train")
    if args.limit is not None:
        src = src.select(range(min(args.limit, len(src))))
    texts, meta = [], []
    for gid, ex in enumerate(src):
        user = [{"role": "user", "content": PROBLEM_PROMPT.format(question=ex["question"])}]
        for gen in ex["generations"]:
            messages = user + [{"role": "assistant", "content": gen}]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
            meta.append((gid, gen))

    out_rows = []
    for start in tqdm(range(0, len(texts), args.batch_size)):
        batch = texts[start : start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=args.max_seq_length).to(device)
        with torch.no_grad():
            log_probs = torch.log_softmax(model(**enc).logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(-1).float().cpu().numpy()
        ids = enc["input_ids"].tolist()
        mask = enc["attention_mask"].tolist()

        for k, (gid, gen) in enumerate(meta[start : start + args.batch_size]):
            comp_start = find_completion_start(ids[k], template_ids)
            if comp_start < 0:
                comp_start = 0
            ent = []
            for t in range(max(comp_start, 1), len(ids[k])):
                if mask[k][t] == 0:
                    break
                ent.append(float(token_entropy[k, t - 1]))
            out_rows.append({
                "messages": [
                    {"role": "user", "content": PROBLEM_PROMPT.format(question=src[gid]["question"])},
                    {"role": "assistant", "content": gen},
                ],
                "score": 0.0,
                "group_id": gid,
                "token_entropies": ent,
            })

    DatasetDict({"train": Dataset.from_list(out_rows)}).push_to_hub(args.output, private=True)
    arr = [np.mean(r["token_entropies"]) for r in out_rows if r["token_entropies"]]
    print(f"{len(out_rows)} rows, mean entropy = {np.mean(arr):.4f}, std = {np.std(arr):.4f}")


if __name__ == "__main__":
    main()
