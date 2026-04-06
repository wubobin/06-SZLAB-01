#!/bin/bash
set -e

ENV_NAME="test_ionic"
PYTHON_VER="3.10"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Creating conda environment: $ENV_NAME"
echo "============================================"

conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "============================================"
echo "  Environment setup complete!"
echo "============================================"
echo ""
echo "Activate:  conda activate $ENV_NAME"
echo "Test:      cd ../test && python test_ionic_predict.py"
echo "SHAP:      cd ../test && python run_ionic_shap.py"
