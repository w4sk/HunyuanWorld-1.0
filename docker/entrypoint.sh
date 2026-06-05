#!/bin/bash
# Activate the conda env, then exec whatever command was passed (default: bash).
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate HunyuanWorld
exec "$@"
