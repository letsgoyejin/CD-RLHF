#!/bin/bash
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate cd_rlhf

mkdir -p CD-RLHF/scripts/logs

pip uninstall flash_attn -y
MAX_JOBS=4 pip install flash-attn --no-build-isolation --force-reinstall \
  &> CD-RLHF/scripts/logs/install_flash_attn.log &

echo $! > CD-RLHF/scripts/logs/flash_attn_install.pid