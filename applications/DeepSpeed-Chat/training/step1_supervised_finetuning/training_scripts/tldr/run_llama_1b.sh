#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate cd_rlhf
# GPU 설정 (GPU 3번 사용)
export CUDA_VISIBLE_DEVICES=3

# huggingface-cli 로그인 (토큰 필요)

# current directory 이동
basepath=/home/soo/yejin/CD-RLHF
cd $basepath/applications/DeepSpeed-Chat/training/step1_supervised_finetuning

# dschat 모듈 경로 추가
export PYTHONPATH=$basepath/applications/DeepSpeed-Chat:$PYTHONPATH

# 모델 저장할 디렉토리 생성
OUTPUT=./models/llama-3.2-1b-tldr-sft
mkdir -p $OUTPUT

deepspeed main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --model_name_or_path meta-llama/Llama-3.2-1B \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 1024 \
   --learning_rate 1e-4 \
   --weight_decay 0.01 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.1 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT/tensorboard \
   --print_loss \
   &> $OUTPUT/training.log
