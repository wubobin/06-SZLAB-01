#!/bin/bash

module purge
module load compiler/gcc/9.3.0
module load compiler/cmake/3.23.3

set -e

ENV_NAME="test_ecw"
PYTHON_VER="3.10"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Creating conda environment: $ENV_NAME"
echo "============================================"

eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
conda activate "$ENV_NAME"
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "============================================"
echo "  Environment setup complete!"
echo "============================================"
echo ""
echo "Activate:  conda activate $ENV_NAME"
echo "Test:      cd ../test && python test_ecw_predict.py && python run_ecw_shap.py"
echo "Train:     cd ../train/scripts && python build_final_20_feature_matrix.py && python compare_rebuilt_descriptors.py && python reproduce_final_20_model.py"
