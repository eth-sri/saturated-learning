import argparse
import itertools
import random

from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompts import PROBLEM_PROMPT, build_judge_prompt, parse_judgment

load_dotenv()


def score_group(samples, llm, tokenizer):
    n = [len(s["generations"]) for s in samples]
    wins = [[0] * k for k in n]
    correct = [[i for i, c in enumerate(s["correct_mask"]) if c] for s in samples]

    comparisons = []
    for sidx, idxs in enumerate(correct):
        for i, j in itertools.combinations(idxs, 2):
            comparisons.append((sidx, i, j))

    if not comparisons:
        return [[1.0 if k == 1 else 0.0 for _ in range(n[s])] for s, k in enumerate(map(len, correct))]

    prompts = []
    for sidx, i, j in comparisons:
        gens = samples[sidx]["generations"]
        prompts.append(build_judge_prompt(samples[sidx]["question"], [gens[i], gens[j]]))

    params = SamplingParams(max_tokens=800, temperature=0, stop=["\n---", "\n## EVALUATION"])
    outputs = llm.generate(prompts, params)

    for (sidx, i, j), out in zip(comparisons, outputs):
        local = parse_judgment(out.outputs[0].text, 2)
        if local is None:
            winner = random.choice([i, j])
        else:
            winner = i if local == 0 else j
        wins[sidx][winner] += 1

    scores = [[0.0] * k for k in n]
    for sidx, idxs in enumerate(correct):
        k = len(idxs)
        if k == 1:
            scores[sidx][idxs[0]] = 1.0
        elif k >= 2:
            for gen_idx in idxs:
                scores[sidx][gen_idx] = wins[sidx][gen_idx]
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge_model", default="Qwen/Qwen3-1.7B-Base")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=None, help="Only judge the first N training samples")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=20480)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(
        model=args.judge_model,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    src = load_dataset(args.dataset, split="train")
    if args.limit is not None:
        src = src.select(range(min(args.limit, len(src))))
    samples = list(src)
    all_scores = score_group(samples, llm, tokenizer)

    rows = []
    for gid, (sample, scores) in enumerate(zip(samples, all_scores)):
        user_msg = {"role": "user", "content": PROBLEM_PROMPT.format(question=sample["question"])}
        for gen, ok, s in zip(sample["generations"], sample["correct_mask"], scores):
            if not ok:
                continue
            rows.append({
                "messages": [user_msg, {"role": "assistant", "content": gen}],
                "score": s,
                "group_id": gid,
            })

    DatasetDict({"train": Dataset.from_list(rows)}).push_to_hub(args.output, private=True)
    print(f"{len(rows)} correct traces across {len(samples)} groups uploaded to {args.output}")


if __name__ == "__main__":
    main()
