#!/bin/bash
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate cd_rlhf
export CUDA_VISIBLE_DEVICES=1

cd CD-RLHF

basepath=/home/soo/yejin/CD-RLHF
LOG_OUTPUT=${basepath}/scripts/logs

python debug.py > ${LOG_OUTPUT}/debug.log 2>&1