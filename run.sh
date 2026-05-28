#!/bin/bash
set -euo pipefail
[[ -f .env ]] && set -a && source .env && set +a

GPU=${1:?gpu index or comma-separated gpu list}
SEED=${2:-42}
IFS=, read -r -a GPU_IDS <<< "$GPU"
TP=${#GPU_IDS[@]}

export CUDA_VISIBLE_DEVICES=$GPU
HF=${HF_NAME:?HF_NAME must be set}
MODEL=Qwen/Qwen3-1.7B-Base
SRC_GENERATIONS=${SRC_GENERATIONS:-$HF/chainsum_generations}
TAG=chainsum_strict_full_s${SEED}
MAX_LEN=2048
JUDGE_MAX_LEN=8192
BASE=outputs/strict_full
EVAL_DATA=$HF/chainsum_eval

ENTROPY_DS=$HF/${TAG}_sw_entropy
INV_ENTROPY_DS=$HF/${TAG}_inv_entropy
ENTROPY_DPO_DS=$HF/${TAG}_entropy_dpo
SJUDGE_DS=$HF/${TAG}_sw
SJUDGE_DPO_DS=$HF/${TAG}_sjudge_dpo

mkdir -p logs "$BASE"

uv run python score_entropy.py \
  --dataset "$SRC_GENERATIONS" \
  --output "$ENTROPY_DS" \
  --max_seq_length "$MAX_LEN" \
  2>&1 | tee logs/${TAG}_score_entropy.log

uv run python score_judge.py \
  --dataset "$SRC_GENERATIONS" \
  --output "$SJUDGE_DS" \
  --max_model_len "$JUDGE_MAX_LEN" \
  --tensor_parallel_size "$TP" \
  2>&1 | tee logs/${TAG}_score_judge.log

uv run python make_data.py inv_entropy --src "$ENTROPY_DS" --out "$INV_ENTROPY_DS"
uv run python make_data.py dpo_pairs --src "$INV_ENTROPY_DS" --out "$ENTROPY_DPO_DS"
uv run python make_data.py dpo_pairs --src "$SJUDGE_DS" --out "$SJUDGE_DPO_DS"

train_dpo() {
  local tag=$1
  local data=$2
  local out=$BASE/${tag}_DPO_s${SEED}
  [[ -d $out && -n "$(find $out -maxdepth 1 -name 'checkpoint-*')" ]] && return
  uv run python -m trl.scripts.dpo \
    --model_name_or_path "$MODEL" --dataset_name "$data" --output_dir "$out" \
    --learning_rate 5e-5 --lr_scheduler_type cosine --warmup_ratio 0.1 \
    --weight_decay 0.0 --optim paged_adamw_8bit --num_train_epochs 1 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
    --gradient_checkpointing --use_peft --lora_r 32 --lora_alpha 64 \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --max_length "$MAX_LEN" \
    --beta 5.0 --loss_type sigmoid \
    --save_strategy steps --save_steps 500 --save_total_limit 2 \
    --seed "$SEED" --run_name ${tag}_DPO_s${SEED} 2>&1 | tee logs/${TAG}_${tag}_DPO.log
}

train_rrhf() {
  local tag=$1
  local data=$2
  local out=$BASE/${tag}_RRHF-BT_s${SEED}
  [[ -d $out && -n "$(find $out -maxdepth 1 -name 'checkpoint-*')" ]] && return
  uv run python train_rrhf.py \
    --model "$MODEL" --dataset "$data" --output_dir "$out" \
    --bt_reweight --rank_weight 0.1 \
    --learning_rate 5e-5 --max_steps 2000 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 1 \
    --lora_r 32 --lora_alpha 64 --max_seq_length "$MAX_LEN" --seed "$SEED" \
    2>&1 | tee logs/${TAG}_${tag}_RRHF-BT.log
}

eval_run() {
  local run=$1
  local out=$BASE/$run
  local eval_out=output/eval/strict_full/$run
  [[ -f $eval_out/summary.json ]] && return
  local ckpt=$(find $out -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)
  [[ -z $ckpt ]] && ckpt=$out
  uv run python eval.py \
    --adapter "$ckpt" \
    --eval_dataset "$EVAL_DATA" \
    --output_dir "$eval_out" \
    --max_tokens "$MAX_LEN" \
    --tensor_parallel_size "$TP" \
    2>&1 | tee logs/${TAG}_${run}_eval.log
}

train_dpo sjudge "$SJUDGE_DPO_DS"
train_dpo invent "$ENTROPY_DPO_DS"
train_rrhf sjudge "$SJUDGE_DS"
train_rrhf invent "$INV_ENTROPY_DS"

eval_run sjudge_DPO_s${SEED} || echo "[warn] sjudge_DPO eval skipped"
eval_run invent_DPO_s${SEED} || echo "[warn] invent_DPO eval skipped"
eval_run sjudge_RRHF-BT_s${SEED}
eval_run invent_RRHF-BT_s${SEED}
