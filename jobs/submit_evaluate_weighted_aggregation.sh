#!/bin/bash
################################################################################
# Evaluate Weighted Probability Aggregation
#
# This script submits a job to evaluate weighted aggregation strategies
# for mapping level 1 probabilities to level 0 Gleason patterns:
#   - Strategy 1: Equal weights (baseline)
#   - Strategy 2: Threshold-based weights (derived from level 1 thresholds)
#   - Strategy 3: Optimized weights (maximizing level 0 Dice on training set)
#
# NOTE: Run evaluate_per_class_thresholds.py first so that
#       threshold_results.json is available for Strategy 2.
#
# Usage:
#   From bsub01.lsf.dkfz.de:
#   bash jobs/submit_evaluate_weighted_aggregation.sh
#
# Configuration:
#   - Checkpoint: /home/f049r/checkpoints/full_data_test/version_8
#   - Data split: [0.9, 0.01, 0.09] (90% train, 1% val, 9% test)
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

# Threshold results path (produced by submit_evaluate_thresholds.sh)
THRESHOLD_RESULTS="${CHECKPOINT_PATH}/threshold_evaluation/threshold_results.json"

# Data split override [train, val, test]
DATA_SPLIT="0.9 0.01 0.09"

# Number of workers
NUM_WORKERS=2

# Timestamp for logs
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

# Derived paths
LOG_DIR="${CHECKPOINT_PATH}/cluster_logs/weighted_aggregation_eval_${TIMESTAMP}"

# Ensure log directory exists
mkdir -p ${LOG_DIR}

# Job parameters
GMEM="20G"
QUEUE="gpu"
NUM_GPUS=1

echo "==========================================="
echo "Evaluate Weighted Probability Aggregation"
echo "==========================================="
echo "User:              ${USER_ID}"
echo "Email:             ${EMAIL}"
echo "Project:           ${PROJECT_DIR}"
echo "Checkpoint:        ${CHECKPOINT_PATH}"
echo "Threshold results: ${THRESHOLD_RESULTS}"
echo "Data split:        ${DATA_SPLIT}"
echo "Num workers:       ${NUM_WORKERS}"
echo "Log directory:     ${LOG_DIR}"
echo ""
echo "Memory request: ${GMEM}"
echo "Queue:          ${QUEUE}"
echo "==========================================="
echo ""

# Build the Python command
PYTHON_CMD="uv run --env-file=.env scripts/evaluate_weighted_aggregation.py \
    --checkpoint_dir ${CHECKPOINT_PATH} \
    --threshold_results_path ${THRESHOLD_RESULTS} \
    --data_split ${DATA_SPLIT} \
    --num_workers ${NUM_WORKERS}"

JOB_NAME="weighted_agg_eval_${TIMESTAMP}"

# Submit job
bsub \
  -gpu "num=${NUM_GPUS}:j_exclusive=yes:gmem=${GMEM}" \
  -R "span[hosts=1]" \
  -R "tensorcore" \
  -q ${QUEUE} \
  -u "${EMAIL}" \
  -N \
  -J "${JOB_NAME}" \
  -o "${LOG_DIR}/weighted_agg_eval_%J.out" \
  -e "${LOG_DIR}/weighted_agg_eval_%J.err" \
  /bin/bash -l -c "source ~/.bashrc && cd ${PROJECT_DIR} && ${PYTHON_CMD}"

if [ $? -eq 0 ]; then
    echo "✓ Weighted aggregation evaluation job submitted!"
    echo ""
    echo "Monitoring commands:"
    echo "  bjobs                    # Check job status"
    echo "  bjobs -w                 # Detailed job status"
    echo "  bpeek <JOB_ID>          # View current output"
    echo "  bkill <JOB_ID>          # Cancel job"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/weighted_agg_eval_<JOB_ID>.out"
    echo "  tail -f ${LOG_DIR}/weighted_agg_eval_<JOB_ID>.err"
    echo ""
    echo "Output will be saved to:"
    echo "  ${CHECKPOINT_PATH}/weighted_aggregation_evaluation/"
else
    echo "✗ Job submission failed!"
    echo "Check that you are on bsub01.lsf.dkfz.de"
    exit 1
fi
