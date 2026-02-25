#!/bin/bash
################################################################################
# GleasonXAI 1-GPU Job Submission Script
#
# Usage:
#   From bsub01.lsf.dkfz.de:
#   bash jobs/submit_gleason_1gpu.sh [HYDRA_OVERRIDES...]
#
# Arguments:
#   HYDRA_OVERRIDES  Any number of Hydra config overrides
#                    These will be passed directly to the training script
#
# Examples:
#   bash jobs/submit_gleason_1gpu.sh dataset.label_level=1 loss_functions=soft_dice_balanced experiment=full_train_test
#   bash jobs/submit_gleason_1gpu.sh experiment=test_run
#   bash jobs/submit_gleason_1gpu.sh dataset.label_level=2 model.learning_rate=0.001
################################################################################

# Configuration - UPDATE THESE VALUES
USER_ID="f049r"
DEPARTMENT="OE0601"
EMAIL="friedrich.heitzer@dkfz-heidelberg.de"
PROJECT_DIR="/home/${USER_ID}/src/ProQuant-AI/GleasonXAI"

# All arguments are Hydra overrides
HYDRA_OVERRIDES="$@"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

# Derived paths
CHECKPOINT_DIR="/home/${USER_ID}/src/ProQuant-AI/logs_gleasonxai"
LOG_DIR="${CHECKPOINT_DIR}/cluster_logs/${TIMESTAMP}"

# Ensure log directory exists
mkdir -p ${LOG_DIR}

# Job parameters
NUM_GPUS=1
GMEM="39G"
QUEUE="gpu-pro"          # gpu queue for general GPU access

# Job name
JOB_NAME="gleason_1gpu_${TIMESTAMP}"

echo "==========================================="
echo "GleasonXAI 1-GPU Job Submission"
echo "==========================================="
echo "User:           ${USER_ID}"
echo "Email:          ${EMAIL}"
echo "Project:        ${PROJECT_DIR}"
if [ -n "${HYDRA_OVERRIDES}" ]; then
    echo "Overrides:      ${HYDRA_OVERRIDES}"
fi
echo "Run ID:         ${TIMESTAMP}"
echo "Log directory:  ${LOG_DIR}"
echo ""
echo "GPU request:    ${NUM_GPUS} GPU @ ${GMEM}"
echo "Queue:          ${QUEUE}"
echo "==========================================="
echo ""
echo "Submitting job..."

# Build the python command with uv
PYTHON_CMD="HYDRA_FULL_ERROR=1 uv run --env-file=.env scripts/run_training.py ${HYDRA_OVERRIDES}"

# Submit the job
bsub \
  -gpu "num=${NUM_GPUS}:j_exclusive=yes:gmem=${GMEM}" \
  -q ${QUEUE} \
  -R "tensorcore" \
  -R "span[hosts=1]" \
  -u "${EMAIL}" \
  -N \
  -J "${JOB_NAME}" \
  -o "${LOG_DIR}/gleason_%J.out" \
  -e "${LOG_DIR}/gleason_%J.err" \
  /bin/bash -l -c "source ~/.bashrc && cd ${PROJECT_DIR} && ${PYTHON_CMD}"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Job submitted successfully!"
    echo ""
    echo "Monitoring commands:"
    echo "  bjobs                    # Check job status"
    echo "  bjobs -w                 # Detailed job status"
    echo "  bpeek <JOB_ID>          # View current output"
    echo "  bprof <JOB_ID>          # Check resource usage"
    echo "  bkill <JOB_ID>          # Cancel job"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOG_DIR}/gleason_<JOB_ID>.out"
    echo "  tail -f ${LOG_DIR}/gleason_<JOB_ID>.err"
    echo ""
    echo "GPU monitoring: https://gpu-monitor.lsf.dkfz.de/"
    echo ""
else
    echo ""
    echo "✗ Job submission failed!"
    echo "Check that you are on bsub01.lsf.dkfz.de"
    exit 1
fi

