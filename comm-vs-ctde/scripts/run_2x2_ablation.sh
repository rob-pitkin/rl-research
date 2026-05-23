#!/bin/bash
# 2x2 ablation matrix for comm-vs-ctde research:
#   {iql, vdn} x {no-comm, with-comm} x {seeds 1..5} = 20 runs total
#
# Runs 2 in parallel by default. Total wall-clock ~4 hours on M1.
# All runs go to wandb project "comm-vs-ctde-final" for clean analysis.
#
# Usage:
#   bash scripts/run_2x2_ablation.sh           # full matrix
#   PARALLEL=1 bash scripts/run_2x2_ablation.sh  # serial (slower but safer)
#   bash scripts/run_2x2_ablation.sh --dry-run   # print commands, don't run

set -u  # error on unset vars

# Resolve project root regardless of where the script is invoked from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- config ----
ALGORITHMS=("iql" "vdn")
COMM_CONDITIONS=("no_comm" "with_comm")
SEEDS=(1 2 3 4 5)
PARALLEL="${PARALLEL:-2}"
WANDB_PROJECT="comm-vs-ctde-final"
LOG_DIR="logs/matrix"

# EPyMARL-canonical hyperparams (locked from seed-1 characterization)
NUM_TIMESTEPS=1000000
EPISODE_LENGTH=25
LR=3e-4
GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=5000
MIN_BUFFER_SIZE=100
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_ANNEAL=200000
TARGET_UPDATE=5000
EVAL_EPISODES=30
EVAL_INTERVAL=50000

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$LOG_DIR"

# ---- build job list ----
JOBS=()
for algo in "${ALGORITHMS[@]}"; do
  for comm in "${COMM_CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOBS+=("${algo}|${comm}|${seed}")
    done
  done
done

echo "==============================================="
echo "  2x2 ablation matrix"
echo "==============================================="
echo "  algorithms     : ${ALGORITHMS[*]}"
echo "  comm conditions: ${COMM_CONDITIONS[*]}"
echo "  seeds          : ${SEEDS[*]}"
echo "  total jobs     : ${#JOBS[@]}"
echo "  parallelism    : $PARALLEL"
echo "  env steps/run  : $NUM_TIMESTEPS"
echo "  wandb project  : $WANDB_PROJECT"
echo "  log dir        : $LOG_DIR"
echo "==============================================="

# ---- launcher for a single job ----
run_job() {
  local spec="$1"
  IFS='|' read -r algo comm seed <<< "$spec"

  local name="${algo}_${comm}_seed${seed}"
  local log_path="$LOG_DIR/${name}.log"

  # Comm flag: no-comm conditions get --mask_communication
  local mask_flag=""
  [[ "$comm" == "no_comm" ]] && mask_flag="--mask_communication"

  # Isolate wandb dir per run to avoid concurrent-run state collisions
  local wandb_dir="wandb/matrix/${name}"
  mkdir -p "$wandb_dir"

  echo "[start] $name"

  WANDB_DIR="$wandb_dir" uv run python -u train.py \
    --algorithm "$algo" \
    $mask_flag \
    --experiment_name "$name" \
    --num_timesteps $NUM_TIMESTEPS \
    --episode_length $EPISODE_LENGTH \
    --lr $LR \
    --gamma $GAMMA \
    --batch_size $BATCH_SIZE \
    --buffer_size $BUFFER_SIZE \
    --min_buffer_size $MIN_BUFFER_SIZE \
    --epsilon_start $EPSILON_START \
    --epsilon_end $EPSILON_END \
    --epsilon_anneal_time $EPSILON_ANNEAL \
    --target_update_freq $TARGET_UPDATE \
    --num_eval_episodes $EVAL_EPISODES \
    --eval_interval $EVAL_INTERVAL \
    --seed "$seed" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    > "$log_path" 2>&1

  echo "[done ] $name (exit $?)"
}

export -f run_job
export LOG_DIR NUM_TIMESTEPS EPISODE_LENGTH LR GAMMA BATCH_SIZE BUFFER_SIZE
export MIN_BUFFER_SIZE EPSILON_START EPSILON_END EPSILON_ANNEAL TARGET_UPDATE
export EVAL_EPISODES EVAL_INTERVAL WANDB_PROJECT

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "DRY RUN — jobs that would be launched:"
  printf '  %s\n' "${JOBS[@]}"
  exit 0
fi

# ---- dispatch with bounded parallelism ----
# xargs handles the queue; -P controls concurrency.
START_TIME=$(date +%s)
printf '%s\n' "${JOBS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'run_job "$@"' _ {}
END_TIME=$(date +%s)

echo ""
echo "==============================================="
echo "  All jobs complete"
echo "  elapsed: $((END_TIME - START_TIME))s"
echo "==============================================="
