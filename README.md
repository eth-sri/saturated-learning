<div align="center">
  <h1>Saturated Learning</h1>
  <p>Training and rerunning saturated learning experiments.</p>

  <a href="https://www.python.org/">
    <img alt="Python: >=3.10" src="https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img alt="Managed with uv" src="https://img.shields.io/badge/Package%20manager-uv-5c5c5c.svg">
  </a>
  <a href="./LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</div>

---

## 👋 Overview

This repository contains the scripts needed to score generated traces, convert
scores into preference data, and run DPO or RRHF-style training experiments.

The minimal workflow is:

1. Score traces with entropy and self-judging signals.
2. Convert scored traces into training datasets.
3. Train adapters from the resulting preference or ranked data.

## 🚀 Setup

```bash
uv sync
```

Create a `.env` file with your Hugging Face namespace and credentials:

```bash
HF_NAME=your-huggingface-name
HF_TOKEN=your-huggingface-token
WANDB_API_KEY=your-wandb-api-key
```

## 🧪 Minimal Experiment

The example below assumes `HF_NAME` points to your Hugging Face namespace and
the source dataset has `question`, `generations`, and `correct_mask` columns.

```bash
export MODEL=Qwen/Qwen3-1.7B-Base
export SRC_GENERATIONS=$HF_NAME/chainsum_generations
export TAG=chainsum_minimal_s42
export MAX_LEN=2048
export JUDGE_MAX_LEN=8192

# 1. Score the generated traces.
uv run python score_entropy.py \
  --model "$MODEL" \
  --dataset "$SRC_GENERATIONS" \
  --output "$HF_NAME/${TAG}_sw_entropy" \
  --max_seq_length "$MAX_LEN"

uv run python score_judge.py \
  --judge_model "$MODEL" \
  --dataset "$SRC_GENERATIONS" \
  --output "$HF_NAME/${TAG}_sw" \
  --max_model_len "$JUDGE_MAX_LEN" \
  --tensor_parallel_size 1

# 2. Convert scores into training data.
uv run python make_data.py inv_entropy \
  --src "$HF_NAME/${TAG}_sw_entropy" \
  --out "$HF_NAME/${TAG}_inv_entropy"

uv run python make_data.py dpo_pairs \
  --src "$HF_NAME/${TAG}_inv_entropy" \
  --out "$HF_NAME/${TAG}_entropy_dpo"

uv run python make_data.py dpo_pairs \
  --src "$HF_NAME/${TAG}_sw" \
  --out "$HF_NAME/${TAG}_sjudge_dpo"

# 3. Run a training experiment.
uv run python -m trl.scripts.dpo \
  --model_name_or_path "$MODEL" \
  --dataset_name "$HF_NAME/${TAG}_sjudge_dpo" \
  --output_dir "outputs/${TAG}_sjudge_DPO" \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --use_peft \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_target_modules q_proj k_proj v_proj o_proj \
  --max_length "$MAX_LEN" \
  --beta 5.0 \
  --loss_type sigmoid

uv run python train_rrhf.py \
  --model "$MODEL" \
  --dataset "$HF_NAME/${TAG}_sw" \
  --output_dir "outputs/${TAG}_sjudge_RRHF-BT" \
  --bt_reweight \
  --rank_weight 0.1 \
  --per_device_train_batch_size 16 \
  --lora_r 32 \
  --lora_alpha 64 \
  --max_seq_length "$MAX_LEN"
```

## 🏃 Convenience Script

For the full default run, use `run.sh` with the GPU index or comma-separated GPU
list. The second argument is the seed and defaults to `42`.

```bash
bash run.sh 0 42
```

Outputs are written under `outputs/`, evaluation summaries under `output/eval/`,
and command logs under `logs/`.
