#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate cd_rlhf
# GPU 설정 (GPU 3번 사용)
export CUDA_VISIBLE_DEVICES=1

# huggingface-cli 로그인 (토큰 필요)

# current directory 이동
basepath=/home/soo/yejin/CD-RLHF
cd $basepath/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning

# dschat 모듈 경로 추가
export PYTHONPATH=$basepath/applications/DeepSpeed-Chat:$PYTHONPATH

# 모델 저장할 디렉토리 생성
OUTPUT=$basepath/models/gemma-2b-tldr-rm
mkdir -p $OUTPUT

# 기록
LOG_OUTPUT=$basepath/scripts/logs
PROJECT_NAME=CD_RLHF

deepspeed --master_port 29501 main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --model_name_or_path $basepath/models/gemma-2b-tldr-sft \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1024 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine \
   --warmup_ratio 0.05 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 2 \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_wandb \
   --project_name $PROJECT_NAME \
   --eval_interval 100 \
   &> $LOG_OUTPUT/training2.log