import argparse
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

load_dotenv()


def build_inv_entropy(src, out):
    ds = load_dataset(src, split="train")
    groups = defaultdict(list)
    for i, row in enumerate(ds):
        groups[row["group_id"]].append((i, row))

    rows = []
    for members in groups.values():
        inv = np.array([1.0 / max(np.mean(r["token_entropies"]), 1e-8) for _, r in members])
        for (_, r), s in zip(members, inv):
            rows.append({
                "messages": r["messages"],
                "score": float(s),
                "group_id": r["group_id"],
                "token_entropies": r["token_entropies"],
            })

    DatasetDict({"train": Dataset.from_list(rows)}).push_to_hub(out, private=True)


def build_dpo_pairs(src, out):
    ds = load_dataset(src, split="train")
    groups = defaultdict(list)
    for row in ds:
        groups[row["group_id"]].append((row["score"], row["messages"]))

    examples = []
    for traces in groups.values():
        if len(traces) < 2:
            continue
        best = max(traces, key=lambda t: t[0])
        worst = min(traces, key=lambda t: t[0])
        if best[0] <= worst[0]:
            continue
        examples.append({
            "prompt": [best[1][0]],
            "chosen": [best[1][1]],
            "rejected": [worst[1][1]],
        })

    DatasetDict({"train": Dataset.from_list(examples)}).push_to_hub(out, private=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["inv_entropy", "dpo_pairs"])
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    (build_inv_entropy if a.mode == "inv_entropy" else build_dpo_pairs)(a.src, a.out)
