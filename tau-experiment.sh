#!/bin/bash
#SBATCH -J tau_experiment
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -A class_cse59827694spring2026
#SBATCH --gres=gpu:hl225:1
#SBATCH -c 12
#SBATCH --mem=40G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e errors/%x_%A_%a.err

set -euo pipefail

#########################################
# ROOT DIR (this script is designed for)
#########################################
ROOT_DIR="${HOME}/agent-project"
LOG_DIR="${ROOT_DIR}/logs"
ERROR_DIR="${ROOT_DIR}/errors"
BASE_RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${LOG_DIR}" "${ERROR_DIR}" "${BASE_RESULTS_DIR}"
cd "${ROOT_DIR}"

#########################################
# REQUIRE USER_MODEL_API_BASE (remote)
#########################################
if [ -z "${USER_MODEL_API_BASE:-}" ]; then
  cat <<EOF
ERROR: USER_MODEL_API_BASE is not set.

You must start the USER vLLM server in a separate job, then set:

  export USER_MODEL_API_BASE="http://<user-node>:8007/v1"

before submitting this script.

EOF
  exit 1
fi

echo "Using USER_MODEL_API_BASE=${USER_MODEL_API_BASE}"
echo

#########################################
# EXPERIMENT GRID (for array mode)
#########################################
ENVS=(airline retail)
AGENTS=(act react fc)
MODELS=(
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-8B-Instruct-2507"
  "Qwen/Qwen3-14B-Instruct-2507"
  "Qwen/Qwen3-32B-Instruct-2507"
)
TRIALS=(1 2 3 4 5)

NUM_ENVS=${#ENVS[@]}       # 2
NUM_AGENTS=${#AGENTS[@]}  # 3
NUM_MODELS=${#MODELS[@]}  # 4
NUM_TRIALS=${#TRIALS[@]}  # 5
TOTAL=$((NUM_ENVS * NUM_AGENTS * NUM_MODELS * NUM_TRIALS))  # 120

#########################################
# ARGUMENT / ARRAY HANDLING
#########################################
# Modes:
#   1) Single run (CLI): sbatch tau-experiment.sh <env> <agent> <assist_model> [num_trials]
#   2) Array mode:      sbatch --array=0-119 tau-experiment.sh
#########################################

if [ "$#" -ge 3 ]; then
  # ----- Mode 1: direct arguments (single experiment) -----
  ENV_NAME="$1"
  AGENT_STRAT_INPUT="$2"
  ASSIST_MODEL="$3"
  if [ "$#" -ge 4 ]; then
    NUM_TRIALS_VAL="$4"
  else
    NUM_TRIALS_VAL=5
  fi

  if [[ "$ENV_NAME" != "retail" && "$ENV_NAME" != "airline" ]]; then
    echo "Error: environment must be 'retail' or 'airline', got '$ENV_NAME'"
    exit 1
  fi

  case "$AGENT_STRAT_INPUT" in
    act|react|fc) ;;
    *)
      echo "Error: agent strategy must be one of: act, react, fc (got '$AGENT_STRAT_INPUT')"
      exit 1
      ;;
  esac
else
  # ----- Mode 2: array mode, decode from SLURM_ARRAY_TASK_ID -----
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    cat <<EOF
Usage:
  Single run:
    sbatch tau-experiment.sh <env: retail|airline> <agent: act|react|fc> <assistant_model_id> [num_trials]

  Full sweep (job array, 120 experiments):
    sbatch --array=0-$((TOTAL-1)) tau-experiment.sh
EOF
    exit 1
  fi

  if [[ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]]; then
    echo "Invalid array index ${SLURM_ARRAY_TASK_ID} (max $((TOTAL-1)))"
    exit 1
  fi

  IDX=$SLURM_ARRAY_TASK_ID

  # index over TRIALS (fastest varying)
  TRIAL_IDX=$((IDX % NUM_TRIALS))
  IDX=$((IDX / NUM_TRIALS))

  # then MODELS
  MODEL_IDX=$((IDX % NUM_MODELS))
  IDX=$((IDX / NUM_MODELS))

  # then AGENTS
  AGENT_IDX=$((IDX % NUM_AGENTS))
  IDX=$((IDX / NUM_AGENTS))

  # then ENVS
  ENV_IDX=$((IDX % NUM_ENVS))

  ENV_NAME="${ENVS[$ENV_IDX]}"
  AGENT_STRAT_INPUT="${AGENTS[$AGENT_IDX]}"
  ASSIST_MODEL="${MODELS[$MODEL_IDX]}"
  NUM_TRIALS_VAL="${TRIALS[$TRIAL_IDX]}"
fi

#########################################
# Map AGENT_STRAT_INPUT -> Tau-Bench CLI
#########################################
case "$AGENT_STRAT_INPUT" in
  act)
    AGENT_STRAT_CLI="act"
    ;;
  react)
    AGENT_STRAT_CLI="react"
    ;;
  fc)
    AGENT_STRAT_CLI="tool-calling"
    ;;
  *)
    echo "Error: agent strategy must be one of: act, react, fc (got '$AGENT_STRAT_INPUT')"
    exit 1
    ;;
esac

USER_MODEL="Qwen/Qwen3-32B"  # logical name for tau-bench; served remotely
ASSIST_SAFE="${ASSIST_MODEL//\//_}"

#########################################
# MODEL SIZE (for dir structure)
#########################################
MODEL_SIZE="unknown"
case "$ASSIST_MODEL" in
  *"4B"*)  MODEL_SIZE="4B" ;;
  *"8B"*)  MODEL_SIZE="8B" ;;
  *"14B"*) MODEL_SIZE="14B" ;;
  *"32B"*) MODEL_SIZE="32B" ;;
esac

echo "========================================"
echo "SLURM_JOB_ID:        ${SLURM_JOB_ID:-N/A}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Environment:         $ENV_NAME"
echo "Agent strategy:      $AGENT_STRAT_INPUT (CLI: $AGENT_STRAT_CLI)"
echo "Assistant model:     $ASSIST_MODEL"
echo "User model (fixed):  $USER_MODEL (remote)"
echo "Num trials:          $NUM_TRIALS_VAL"
echo "Model size bucket:   $MODEL_SIZE"
echo "========================================"
echo

#########################################
# RESULTS DIRS (mirror log structure)
# results/env/agent/modelsize/<assist_safe>_trialsN
#########################################
RESULTS_SUBDIR="${BASE_RESULTS_DIR}/${ENV_NAME}/${AGENT_STRAT_INPUT}/${MODEL_SIZE}"
mkdir -p "${RESULTS_SUBDIR}"


echo "Results directory:   $RESULTS_SUBDIR"
echo

#########################################
# LOG DIR STRUCTURE: logs/env/agent/modelsize
#########################################
LOG_SUBDIR="${LOG_DIR}/${ENV_NAME}/${AGENT_STRAT_INPUT}/${MODEL_SIZE}"
mkdir -p "${LOG_SUBDIR}"

echo "Log directory:       ${LOG_SUBDIR}"
echo

#########################################
# ACTIVATE ENV (Gaudi vLLM / PyTorch)
#########################################
module load mamba/latest
source activate gaudi-pytorch-vllm
export NO_AI_TRACKING=true
export VLLM_BUILD="0.0.0.0"

#########################################
# Quick health check on USER server
#########################################
echo "Checking USER model endpoint health..."
if ! curl -s "${USER_MODEL_API_BASE}/models" > /dev/null; then
  echo "ERROR: Could not reach USER_MODEL_API_BASE=${USER_MODEL_API_BASE}/models"
  echo "Make sure the user-vllm job is running and URL is correct."
  exit 1
fi
echo "USER endpoint is reachable."
echo

#########################################
# START ASSISTANT SERVER (LOCAL)
#########################################

echo "Starting ASSISTANT vLLM on port 8005 (node: $(hostname))..."
ASSIST_LOG="${LOG_SUBDIR}/assistant_${ASSIST_SAFE}_trials${NUM_TRIALS_VAL}_job${SLURM_JOB_ID}.log"
./assistant-server.sh "$ASSIST_MODEL" 8005 \
  > "${ASSIST_LOG}" 2>&1 &
ASSIST_PID=$!

echo "Assistant PID: $ASSIST_PID"
echo "Assistant log: ${ASSIST_LOG}"
echo

#########################################
# HELPER: wait until vLLM server is ready
#########################################
wait_for_ready() {
  local name="$1"
  local base="$2"
  local max_attempts=240
  local attempt=1

  echo "Waiting for ${name} server at ${base}/models to become ready..."

  while (( attempt <= max_attempts )); do
    if curl -s "${base}/models" > /dev/null; then
      echo "${name} server at ${base} is ready (after ${attempt} attempts)."
      return 0
    fi

    echo "  [${name}] not ready yet (attempt ${attempt}/${max_attempts}). Sleeping 5s..."
    sleep 5
    ((attempt++))
  done

  echo "ERROR: ${name} server at ${base} did not become ready in time."
  return 1
}

#########################################
# WAIT FOR ASSISTANT TO BE READY
#########################################
ASSIST_BASE="http://127.0.0.1:8005/v1"

if ! wait_for_ready "ASSISTANT" "$ASSIST_BASE"; then
  echo "Assistant server failed to start correctly. Cleaning up and exiting."
  kill "$ASSIST_PID" 2>/dev/null || true
  wait "$ASSIST_PID" 2>/dev/null || true
  exit 1
fi

#########################################
# EXPORT ENDPOINTS
#########################################
export OPENAI_API_BASE="${ASSIST_BASE}"         # assistant (local)
export OPENAI_API_KEY="EMPTY"

#########################################
# RUN τ-BENCH
#########################################
cd "${ROOT_DIR}/tau-bench"

echo "Running Tau-Bench..."
echo

# Usage: set START_INDEX for resume; call before the python run.
# Updated to:
#   - If any .info.error exists -> resume from FIRST error task_id
#   - Else -> resume from (max task_id + 1)
set_start_index_from_checkpoint() {
  local ckpt="${RESULTS_SUBDIR}/num_trials-${NUM_TRIALS_VAL}.json"
  START_INDEX=0

  if [[ -f "$ckpt" ]]; then
    # first task_id with a non-empty .info.error, if any
    local error_start
    error_start=$(jq -r '[.[] | select(.info.error != null and .info.error != "") | .task_id] | min // empty' "$ckpt" 2>/dev/null || true)

    # max task_id overall
    local max_id
    max_id=$(jq -r '[.[].task_id] | max // empty' "$ckpt" 2>/dev/null || true)

    if [[ -n "$error_start" && "$error_start" != "null" ]]; then
      # We found at least one errored task; restart from the earliest one
      START_INDEX="$error_start"
      echo "Resuming: found errored tasks, restarting from first error task_id=$START_INDEX"
    elif [[ -n "$max_id" && "$max_id" != "null" ]]; then
      # No errors; continue after last successful task
      START_INDEX=$((max_id + 1))
      echo "Resuming: last completed task_id=$max_id, --start-index=$START_INDEX"
    else
      echo "Checkpoint exists but could not determine task_ids; starting from 0."
      START_INDEX=0
    fi
  else
    echo "No checkpoint file at $ckpt; starting from first task (start-index=0)."
  fi
}

set_start_index_from_checkpoint

python run.py \
  --agent-strategy "$AGENT_STRAT_CLI" \
  --env "$ENV_NAME" \
  --model "$ASSIST_MODEL" \
  --model-provider openai \
  --user-model "$USER_MODEL" \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 0.6 \
  --start-index "$START_INDEX" \
  --end-index -1 \
  --max-concurrency 1 \
  --num-trials "$NUM_TRIALS_VAL" \
  --log-dir "$RESULTS_SUBDIR" \

TB_EXIT=$?

#########################################
# CLEANUP
#########################################
echo "Shutting down assistant vLLM (PID: $ASSIST_PID)"
kill "$ASSIST_PID" 2>/dev/null || true
wait "$ASSIST_PID" 2>/dev/null || true

echo "Tau-Bench finished with exit code: $TB_EXIT"
exit "$TB_EXIT"
