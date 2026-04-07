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
OUTPUT=$basepath/models/gemma-2b-tldr-sft
mkdir -p $OUTPUT

# 기록
LOG_OUTPUT=$basepath/scripts/logs
PROJECT_NAME=CD_RLHF

deepspeed --master_port 29501 main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --model_name_or_path google/gemma-2b \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 2048 \
   --learning_rate 5e-5 \
   --weight_decay 0.01 \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 32 \
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
   --enable_wandb \
   --project_name $PROJECT_NAME \
   &> $LOG_OUTPUT/training1.log &

echo $! > $LOG_OUTPUT/gemma-2b-tldr-sft.pid
echo "Training started with PID: $!"