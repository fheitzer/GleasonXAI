#!/bin/bash
################################################################################
# Evaluate Per-Class Thresholds
#
# This script submits a job to evaluate per-class thresholds:
#   - Optimizes thresholds on training set
#   - Evaluates on both training and test sets
#   - Generates visualizations and metrics
#
# Usage:
#   From bsub01.lsf.dkfz.de:
#   bash jobs/submit_evaluate_thresholds.sh
#
# Configuration:
#   - Checkpoint: /home/f049r/checkpoints/full_data_test/version_8
#   - Data split: [0.9, 0.01, 0.09] (90% train, 1% val, 9% test)
#   - Metric: dice
################################################################################

# Configuration
USER_ID="f049r"
DEPARTMENT="OE0601"
EMAIL="friedrich.heitzer@dkfz-heidelberg.de"
PROJECT_DIR="/home/${USER_ID}/src/ProQuant-AI/GleasonXAI"

# Checkpoint configuration
CHECKPOINT_BASE="/home/${USER_ID}/checkpoints/full_data_test"
CHECKPOINT_VERSION="version_8"
CHECKPOINT_PATH="${CHECKPOINT_BASE}/${CHECKPOINT_VERSION}"

# Data split override [train, val, test]
DATA_SPLIT="0.9 0.01 0.09"

# Optimization metric
METRIC="dice"

# Number of workers (use 0 to avoid memory issues)
NUM_WORKERS=2

# Timestamp for logs
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

# Derived paths
LOG_DIR="${CHECKPOINT_PATH}/cluster_logs/threshold_eval_${TIMESTAMP}"

# Ensure log directory exists
mkdir -p ${LOG_DIR}

# Job parameters (CPU only, no GPU needed)
GMEM="20G"
QUEUE="gpu"
NUM_GPUS=1

echo "==========================================="
echo "Evaluate Per-Class Thresholds"
echo "==========================================="
echo "User:           ${USER_ID}"
echo "Email:          ${EMAIL}"
echo "Project:        ${PROJECT_DIR}"
echo "Checkpoint:     ${CHECKPOINT_PATH}"
echo "Data split:     ${DATA_SPLIT}"
echo "Metric:         ${METRIC}"
echo "Num workers:    ${NUM_WORKERS}"
echo "Log directory:  ${LOG_DIR}"
echo ""
echo "Memory request: ${MEM}"
echo "CPU cores:      ${NUM_CORES}"
echo "Queue:          ${QUEUE}"
echo "==========================================="
echo ""

# Build the Python command
PYTHON_CMD="uv run --env-file=.env scripts/evaluate_per_class_thresholds.py \
    --checkpoint_dir ${CHECKPOINT_PATH} \
    --data_split ${DATA_SPLIT} \
    --metric ${METRIC} \
    --num_workers ${NUM_WORKERS}"

JOB_NAME="thresh_eval_${TIMESTAMP}"

# Submit job
bsub \
  -gpu "num=${NUM_GPUS}:j_exclusive=yes:gmem=${GMEM}" \
  -R "span[hosts=1]" \
  -R "tensorcore" \
  -q ${QUEUE} \
  -u "${EMAIL}" \
  -N \
  -J "${JOB_NAME}" \
  -o "${LOG_DIR}/threshold_eval_%J.out" \
  -e "${LOG_DIR}/threshold_eval_%J.err" \
  /bin/bash -l -c "source ~/.bashrc && cd ${PROJECT_DIR} && ${PYTHON_CMD}"

if [ $? -eq 0 ]; then
    echo "✓ Threshold evaluation job submitted!"
    echo ""
    echo "Monitoring commands:"
    echo "  bjobs                    # Check job status"
    echo "  bjobs -w                 # Detailed job status"
    echo "  bpeek <JOB_ID>          # View current output"
    echo "  bkill <JOB_ID>          # Cancel job"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/threshold_eval_<JOB_ID>.out"
    echo "  tail -f ${LOG_DIR}/threshold_eval_<JOB_ID>.err"
    echo ""
    echo "Output will be saved to:"
    echo "  ${CHECKPOINT_PATH}/threshold_evaluation/"
else
    echo "✗ Job submission failed!"
    echo "Check that you are on bsub01.lsf.dkfz.de"
    exit 1
fi
