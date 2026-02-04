#!/bin/bash
################################################################################
# Evaluate Level 0 (Gleason Pattern) Metrics
#
# This script submits a job to evaluate level 0 metrics:
#   - Remaps level 1 predictions to level 0 (Gleason patterns)
#   - Evaluates with default 0.5 threshold (for comparison with paper)
#   - Optionally optimizes thresholds on level 0 training set
#   - Generates metrics and comparison with paper results
#
# Usage:
#   From bsub01.lsf.dkfz.de:
#   bash jobs/submit_evaluate_level0.sh
#
# Configuration:
#   - Checkpoint: /home/f049r/checkpoints/full_data_test/version_8
#   - Evaluates level 0 (4 classes: Benign, Pattern 3, Pattern 4, Pattern 5)
#   - Compares with paper's reported 0.713 Dice
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

# Whether to optimize thresholds on level 0 (add --optimize_thresholds flag)
# NOTE: Threshold optimization requires loading 914 train labels which uses ~91GB RAM
# For now, disable this to just compare with paper using default 0.5 threshold
OPTIMIZE_THRESHOLDS=true

# Number of workers (use 0 to avoid memory issues)
NUM_WORKERS=10

# Timestamp for logs
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

# Derived paths
LOG_DIR="${CHECKPOINT_PATH}/cluster_logs/level0_eval_${TIMESTAMP}"

# Ensure log directory exists
mkdir -p ${LOG_DIR}

# Job parameters
GMEM="20G"  # GPU memory
QUEUE="gpu-pro"
NUM_GPUS=1

echo "==========================================="
echo "Evaluate Level 0 (Gleason Pattern) Metrics"
echo "==========================================="
echo "User:           ${USER_ID}"
echo "Email:          ${EMAIL}"
echo "Project:        ${PROJECT_DIR}"
echo "Checkpoint:     ${CHECKPOINT_PATH}"
echo "Optimize thresholds: ${OPTIMIZE_THRESHOLDS}"
echo "Num workers:    ${NUM_WORKERS}"
echo "Log directory:  ${LOG_DIR}"
echo ""
echo "Memory request: ${MEM}"
echo "CPU cores:      ${NUM_CORES}"
echo "Queue:          ${QUEUE}"
echo "==========================================="
echo ""

# Build the Python command
PYTHON_CMD="uv run --env-file=.env scripts/evaluate_level0_metrics.py \
    --checkpoint_dir ${CHECKPOINT_PATH}"

# Add optimize_thresholds flag if enabled
if [ "${OPTIMIZE_THRESHOLDS}" = true ]; then
    PYTHON_CMD="${PYTHON_CMD} --optimize_thresholds"
fi

JOB_NAME="level0_eval_${TIMESTAMP}"

# Submit job
bsub \
  -gpu "num=${NUM_GPUS}:j_exclusive=yes:gmem=${GMEM}" \
  -R "span[hosts=1]" \
  -R "tensorcore" \
  -q ${QUEUE} \
  -u "${EMAIL}" \
  -N \
  -J "${JOB_NAME}" \
  -o "${LOG_DIR}/level0_eval_%J.out" \
  -e "${LOG_DIR}/level0_eval_%J.err" \
  /bin/bash -l -c "source ~/.bashrc && cd ${PROJECT_DIR} && ${PYTHON_CMD}"

if [ $? -eq 0 ]; then
    echo "✓ Level 0 evaluation job submitted!"
    echo ""
    echo "Monitoring commands:"
    echo "  bjobs                    # Check job status"
    echo "  bjobs -w                 # Detailed job status"
    echo "  bpeek <JOB_ID>          # View current output"
    echo "  bkill <JOB_ID>          # Cancel job"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/level0_eval_<JOB_ID>.out"
    echo "  tail -f ${LOG_DIR}/level0_eval_<JOB_ID>.err"
    echo ""
    echo "Output will be saved to:"
    echo "  ${CHECKPOINT_PATH}/level0_evaluation/"
else
    echo "✗ Job submission failed!"
    echo "Check that you are on bsub01.lsf.dkfz.de"
    exit 1
fi
