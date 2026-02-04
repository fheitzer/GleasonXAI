#!/bin/bash
################################################################################
# Generate Predictions Locally (No Cluster)
#
# This script generates predictions locally for threshold optimization.
# Use this for testing or if you have a GPU available locally.
#
# Usage:
#   bash jobs/generate_predictions_local.sh
################################################################################

# Configuration
PROJECT_DIR="/home/f049r/src/ProQuant-AI/GleasonXAI"
CHECKPOINT_BASE="/home/f049r/checkpoints/full_data_test"
CHECKPOINT_VERSION="version_8"
CHECKPOINT_PATH="${CHECKPOINT_BASE}/${CHECKPOINT_VERSION}"

echo "==========================================="
echo "Generate Predictions Locally"
echo "==========================================="
echo "Project:        ${PROJECT_DIR}"
echo "Checkpoint:     ${CHECKPOINT_PATH}"
echo "Data split:     [0.9, 0.01, 0.09]"
echo "==========================================="
echo ""

# Change to project directory
cd ${PROJECT_DIR}

# Helper function to create temporary config with overridden data_split
create_temp_config() {
    local ORIGINAL_CONFIG="${CHECKPOINT_PATH}/logs/config.yaml"
    local TEMP_CONFIG="${CHECKPOINT_PATH}/logs/config_temp_${1}.yaml"

    echo "Creating temporary config for ${1}..."

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

print(f"✓ Created temporary config with data_split=[0.9, 0.01, 0.09]")
EOF

    echo "${TEMP_CONFIG}"
}

# ==============================================================================
# Generate TRAINING predictions
# ==============================================================================
echo ""
echo "Step 1/2: Generating TRAINING predictions..."
echo "=============================================="

TEMP_CONFIG_TRAIN=$(create_temp_config "train")

uv run --env-file=.env scripts/test.py \
    --experiment_path ${CHECKPOINT_BASE} \
    --checkpoint ${CHECKPOINT_VERSION} \
    --config ${TEMP_CONFIG_TRAIN} \
    --task predict \
    --eval_on train \
    --no_logging

if [ $? -eq 0 ]; then
    echo "✓ Training predictions generated successfully!"
    echo "  Saved to: ${CHECKPOINT_PATH}/preds/pred_train.pt"
    TRAIN_SUCCESS=true
else
    echo "✗ Failed to generate training predictions!"
    TRAIN_SUCCESS=false
fi

# Clean up temporary config
rm -f ${TEMP_CONFIG_TRAIN}

# ==============================================================================
# Generate TEST predictions
# ==============================================================================
echo ""
echo "Step 2/2: Generating TEST predictions..."
echo "=========================================="

TEMP_CONFIG_TEST=$(create_temp_config "test")

uv run --env-file=.env scripts/test.py \
    --experiment_path ${CHECKPOINT_BASE} \
    --checkpoint ${CHECKPOINT_VERSION} \
    --config ${TEMP_CONFIG_TEST} \
    --task predict \
    --eval_on test \
    --no_logging

if [ $? -eq 0 ]; then
    echo "✓ Test predictions generated successfully!"
    echo "  Saved to: ${CHECKPOINT_PATH}/preds/pred_test.pt"
    TEST_SUCCESS=true
else
    echo "✗ Failed to generate test predictions!"
    TEST_SUCCESS=false
fi

# Clean up temporary config
rm -f ${TEMP_CONFIG_TEST}

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "==========================================="
if [ "$TRAIN_SUCCESS" = true ] && [ "$TEST_SUCCESS" = true ]; then
    echo "✓ All predictions generated successfully!"
    echo "==========================================="
    echo ""
    echo "Output files:"
    echo "  Training: ${CHECKPOINT_PATH}/preds/pred_train.pt"
    echo "  Test:     ${CHECKPOINT_PATH}/preds/pred_test.pt"
    echo ""
    echo "Next step: Run threshold optimization"
    echo ""
    echo "  python scripts/evaluate_per_class_thresholds.py \\"
    echo "      --checkpoint_dir ${CHECKPOINT_PATH} \\"
    echo "      --data_split 0.9 0.01 0.09 \\"
    echo "      --metric dice"
    echo ""
elif [ "$TRAIN_SUCCESS" = true ]; then
    echo "⚠ Partial success: Training predictions generated"
    echo "==========================================="
    echo "Test predictions failed. Check the error messages above."
    exit 1
elif [ "$TEST_SUCCESS" = true ]; then
    echo "⚠ Partial success: Test predictions generated"
    echo "==========================================="
    echo "Training predictions failed. Check the error messages above."
    exit 1
else
    echo "✗ Failed to generate predictions!"
    echo "==========================================="
    echo "Check the error messages above."
    exit 1
fi
