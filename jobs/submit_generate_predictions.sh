#!/bin/bash
################################################################################
# Generate Predictions for Threshold Optimization
#
# This script submits 2 jobs:
#   1. Generate training predictions with custom data split
#   2. Generate test predictions with custom data split
#
# Usage:
#   From bsub01.lsf.dkfz.de:
#   bash jobs/submit_generate_predictions.sh
#
# Configuration:
#   - Checkpoint: /home/f049r/checkpoints/full_data_test/version_8
#   - Data split: [0.9, 0.01, 0.09] (90% train, 1% val, 9% test)
#   - Seed: 42 (from training config)
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
DATA_SPLIT="[0.9,0.01,0.09]"

# Timestamp for logs
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

# Derived paths
LOG_DIR="${CHECKPOINT_PATH}/cluster_logs/predictions_${TIMESTAMP}"

# Ensure log directory exists
mkdir -p ${LOG_DIR}

# Job parameters
NUM_GPUS=1
GMEM="39G"
QUEUE="gpu-pro"

echo "==========================================="
echo "Generate Predictions for Threshold Optimization"
echo "==========================================="
echo "User:           ${USER_ID}"
echo "Email:          ${EMAIL}"
echo "Project:        ${PROJECT_DIR}"
echo "Checkpoint:     ${CHECKPOINT_PATH}"
echo "Data split:     ${DATA_SPLIT}"
echo "Log directory:  ${LOG_DIR}"
echo ""
echo "GPU request:    ${NUM_GPUS} GPU @ ${GMEM}"
echo "Queue:          ${QUEUE}"
echo "==========================================="
echo ""

# Helper function to create temporary config with overridden data_split
create_temp_config() {
    local ORIGINAL_CONFIG="${CHECKPOINT_PATH}/logs/config.yaml"
    local TEMP_CONFIG="${CHECKPOINT_PATH}/logs/config_temp_${1}.yaml"

    # Copy original config
    cp "${ORIGINAL_CONFIG}" "${TEMP_CONFIG}"

    # Override data_split using Python
    python3 << EOF
import yaml
from pathlib import Path

config_path = Path("${TEMP_CONFIG}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Override data_split
config['dataset']['data_split'] = [0.9, 0.01, 0.09]

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

import sys
print(f"Created temporary config with data_split=[0.9, 0.01, 0.09]", file=sys.stderr)
EOF

    echo "${TEMP_CONFIG}"
}

# ==============================================================================
# JOB 1: Generate TRAINING predictions
# ==============================================================================
echo "Submitting Job 1: Generate TRAINING predictions..."
echo ""

TEMP_CONFIG_TRAIN=$(create_temp_config "train")
JOB_NAME_TRAIN="pred_train_${TIMESTAMP}"

PYTHON_CMD_TRAIN="uv run --env-file=.env scripts/test.py \
    --experiment_path ${CHECKPOINT_BASE} \
    --checkpoint ${CHECKPOINT_VERSION} \
    --config ${TEMP_CONFIG_TRAIN} \
    --task predict \
    --eval_on train \
    --no_logging"

bsub \
  -gpu "num=${NUM_GPUS}:j_exclusive=yes:gmem=${GMEM}" \
  -q ${QUEUE} \
  -R "tensorcore" \
  -R "span[hosts=1]" \
  -u "${EMAIL}" \
  -N \
  -J "${JOB_NAME_TRAIN}" \
  -o "${LOG_DIR}/pred_train_%J.out" \
  -e "${LOG_DIR}/pred_train_%J.err" \
  /bin/bash -l -c "source ~/.bashrc && cd ${PROJECT_DIR} && ${PYTHON_CMD_TRAIN}"

if [ $? -eq 0 ]; then
    echo "✓ Training predictions job submitted!"
    TRAIN_JOB_SUBMITTED=true
else
    echo "✗ Training predictions job submission failed!"
    TRAIN_JOB_SUBMITTED=false
fi

echo ""

# ==============================================================================
# JOB 2: Generate TEST predictions
# ==============================================================================
echo "Submitting Job 2: Generate TEST predictions..."
echo ""

TEMP_CONFIG_TEST=$(create_temp_config "test")
JOB_NAME_TEST="pred_test_${TIMESTAMP}"

PYTHON_CMD_TEST="uv run --env-file=.env scripts/test.py \
    --experiment_path ${CHECKPOINT_BASE} \
    --checkpoint ${CHECKPOINT_VERSION} \
    --config ${TEMP_CONFIG_TEST} \
    --task predict \
    --eval_on test \
    --no_logging"

bsub \
  -gpu "num=${NUM_GPUS}:j_exclusive=yes:gmem=${GMEM}" \
  -q ${QUEUE} \
  -R "tensorcore" \
  -R "span[hosts=1]" \
  -u "${EMAIL}" \
  -N \
  -J "${JOB_NAME_TEST}" \
  -o "${LOG_DIR}/pred_test_%J.out" \
  -e "${LOG_DIR}/pred_test_%J.err" \
  /bin/bash -l -c "source ~/.bashrc && cd ${PROJECT_DIR} && ${PYTHON_CMD_TEST}"

if [ $? -eq 0 ]; then
    echo "✓ Test predictions job submitted!"
    TEST_JOB_SUBMITTED=true
else
    echo "✗ Test predictions job submission failed!"
    TEST_JOB_SUBMITTED=false
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================
if [ "$TRAIN_JOB_SUBMITTED" = true ] && [ "$TEST_JOB_SUBMITTED" = true ]; then
    echo "==========================================="
    echo "✓ Both jobs submitted successfully!"
    echo "==========================================="
    echo ""
    echo "Output files will be saved to:"
    echo "  Training: ${CHECKPOINT_PATH}/preds/pred_train.pt"
    echo "  Test:     ${CHECKPOINT_PATH}/preds/pred_test.pt"
    echo ""
    echo "Monitoring commands:"
    echo "  bjobs                    # Check job status"
    echo "  bjobs -w                 # Detailed job status"
    echo "  bpeek <JOB_ID>          # View current output"
    echo "  bkill <JOB_ID>          # Cancel job"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/pred_train_<JOB_ID>.out"
    echo "  tail -f ${LOG_DIR}/pred_test_<JOB_ID>.out"
    echo ""
    echo "After both jobs complete, run threshold optimization:"
    echo "  python scripts/evaluate_per_class_thresholds.py \\"
    echo "      --checkpoint_dir ${CHECKPOINT_PATH} \\"
    echo "      --data_split 0.9 0.01 0.09 \\"
    echo "      --metric dice"
    echo ""
    echo "GPU monitoring: https://gpu-monitor.lsf.dkfz.de/"
    echo ""
else
    echo "==========================================="
    echo "✗ Job submission failed!"
    echo "==========================================="
    echo "Check that you are on bsub01.lsf.dkfz.de"
    exit 1
fi
