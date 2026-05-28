import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from prompts import PROBLEM_PROMPT
from verifiers import verify

load_dotenv()


def merge_lora(base, adapter, tokenizer):
    out = tempfile.mkdtemp(prefix="merged_")
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", trust_remote_code=True)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.save_pretrained(out)
    tokenizer.save_pretrained(out)
    del m
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--adapter", default=None)
    p.add_argument("--eval_dataset", required=True)
    p.add_argument("--n_questions", type=int, default=200)
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    merged = merge_lora(args.base_model, args.adapter, tokenizer) if args.adapter else args.base_model

    llm = LLM(
        model=merged,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    ds = load_dataset(args.eval_dataset, split="train").select(range(args.n_questions))
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROBLEM_PROMPT.format(question=ex["question"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in ds
    ]

    outs = llm.generate(
        prompts,
        SamplingParams(n=args.n_samples, temperature=args.temperature, max_tokens=args.max_tokens, top_p=0.95),
    )

    total_avg, total_pass, records = 0.0, 0, []
    with open(out_dir / "generations.jsonl", "w") as f:
        for ex, out in zip(ds, outs):
            gens = [o.text for o in out.outputs]
            gt = str(ex["ground_truth"]).split("####")[-1].strip()
            results = [verify(g, gt) for g in gens]
            correct = sum(1 for _, ok in results if ok)
            avg = correct / args.n_samples
            pass_at = int(correct > 0)
            total_avg += avg
            total_pass += pass_at
            f.write(json.dumps({
                "question": ex["question"],
                "ground_truth": gt,
                "generations": gens,
                "extracted": [a for a, _ in results],
                "correct_mask": [ok for _, ok in results],
            }) + "\n")

    n = len(ds)
    summary = {
        "model": args.base_model,
        "adapter": args.adapter,
        "n_samples": args.n_samples,
        f"avg_at_{args.n_samples}": total_avg / n,
        f"pass_at_{args.n_samples}": total_pass / n,
        "total_questions": n,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"avg@{args.n_samples}={summary[f'avg_at_{args.n_samples}']:.4f}  "
          f"pass@{args.n_samples}={summary[f'pass_at_{args.n_samples}']:.4f}")

    if args.adapter:
        shutil.rmtree(merged, ignore_errors=True)


if __name__ == "__main__":
    main()
