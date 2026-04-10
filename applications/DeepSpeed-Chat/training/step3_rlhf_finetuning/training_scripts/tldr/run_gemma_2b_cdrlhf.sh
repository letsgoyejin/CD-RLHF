#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate cd_rlhf2
# GPU 설정 (GPU 2,3번 사용)
GPU_ID=2,3
export CUDA_VISIBLE_DEVICES=$GPU_ID

# huggingface-cli 로그인 (토큰 필요)

# current directory 이동
basepath=/home/soo/yejin/CD-RLHF
cd $basepath/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning

# dschat 모듈 경로 추가
export PYTHONPATH=$basepath/applications/DeepSpeed-Chat:$PYTHONPATH

# 메모리 단편화 줄이기
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cuda 버전 체크 건너뛰기
export DS_SKIP_CUDA_CHECK=1

# 모델 저장할 디렉토리 생성
OUTPUT=$basepath/models/gemma-2b-tldr-cdrlhf
mkdir -p $OUTPUT

# 기록
LOG_OUTPUT=$basepath/scripts/logs
PROJECT_NAME=CD_RLHF

echo $(basename $OUTPUT)
branch_info=$(git branch | grep '*')
commit_info=$(git rev-parse --short HEAD)
echo "branch: $branch_info commit id: $commit_info" > $OUTPUT/training.log

Actor_Lr=8e-6
Critic_Lr=1e-5

# 메모리 모니터링 시작 (백그라운드)
> "$LOG_OUTPUT/memory_log.txt"
while true; do
    echo "=== $(date) ===" >> $LOG_OUTPUT/memory_log.txt
    nvidia-smi -i $GPU_ID --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader >> $LOG_OUTPUT/memory_log.txt
    free -h >> $LOG_OUTPUT/memory_log.txt
    sleep 10
done &
MONITOR_PID=$!

# 스크립트 종료 시 (정상/비정상 모두) 모니터링 종료
trap "kill $MONITOR_PID 2>/dev/null" EXIT

deepspeed --master_port 29502 --include localhost:$GPU_ID main.py \
   --data_path openai/summarize_from_feedback \
   --data_split 2,4,4 \
   --actor_model_name_or_path $basepath/models/gemma-2b-tldr-sft \
   --critic_model_name_or_path $basepath/models/gemma-2b-tldr-rm/step_18572 \
   --dtype bf16 \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 1 \
   --per_device_training_batch_size 1 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type linear \
   --gradient_accumulation_steps 32 \
   --end_of_conversation_token "<eos>" \
   --actor_dropout 0.0 \
   --warmup_ratio 0.1 \
   --deepspeed --seed 1234 \
   --actor_zero_stage 2 \
   --critic_zero_stage 2 \
   --output_dir $OUTPUT \
   --icm_learning_rate 1e-5 \
   --eta 0.04 \
   --cdrlhf_topk 1 \
   --sample_size 100 \
   --kl_ctl 0.05 \
   --print_answers \
   --print_answers_interval 100 \
   --save_steps 1000 \
   --min_new_tokens 4 \
   --enable_wandb \
   --project_name $PROJECT_NAME \
    &> $LOG_OUTPUT/training3.log
