#!/bin/bash
set -e

MODELS=(

# 'DLinear'
# 'Transformer'
'LSTM'
# 'iTransformer'
# 'GRU'
# 'Nonstationary_Transformer'

)

DATASET_NAME=(
    "mophong_data_ITS"
    # "usdvnd"
)

TARGET="Low,High"

ROOT_PATH="$(dirname "$(dirname "$(realpath "$0")")")/dataset"


VENV_DIR="$(dirname "$(dirname "$(realpath "$0")")")/.venv"
LOG_DIR="./logs/ONE&ONLY"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at: $VENV_DIR"
    exit 1
fi
source "$VENV_DIR/bin/activate"

mkdir -p "$LOG_DIR"

# -------------------- LOOP THROUGH STATIONS AND MODELS -------------------- #
for MODEL_NAME in "${MODELS[@]}"; do
        # DATASET_NAME="${STATION}"
        echo ""
        echo "Starting Optuna tuning for model: $MODEL_NAME on data: $STATION"
        START_TIME=$(date +%s)

        LOG_FILE="${LOG_DIR}/optuna_${MODEL_NAME}_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"
        echo "Logging to: $LOG_FILE"

        python -u run_optun.py \
          --is_training 1 \
          --root_path "$ROOT_PATH" \
          --data "custom"\
          --data_path "${DATASET_NAME}.csv" \
          --model_id "$FILENAME" \
          --model $MODEL_NAME \
          --target $TARGET \
          --task_name short_term_forecast\
            2>&1 | tee "$LOG_FILE"

        END_TIME=$(date +%s)
        ELAPSED_TIME=$((END_TIME - START_TIME))

        echo ""
        echo "Optuna tuning completed in $ELAPSED_TIME seconds."
        echo "Full log saved at: $LOG_FILE"
done
