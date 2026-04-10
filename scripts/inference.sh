#!/bin/bash
# conda 환경 활성화
source /home/soo/miniconda3/etc/profile.d/conda.sh
conda activate cd_rlhf2
# GPU 설정 (GPU 0,1번 사용)
GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID

# current directory 이동
basepath=/home/soo/yejin/CD-RLHF
cd $basepath

# dschat 모듈 경로 추가
export PYTHONPATH=$basepath/applications/DeepSpeed-Chat:$PYTHONPATH

# 메모리 단편화 줄이기
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cuda 버전 체크 건너뛰기
export DS_SKIP_CUDA_CHECK=1

# 기록
LOG_OUTPUT=$basepath/scripts/logs
exec > $LOG_OUTPUT/gemma-2b-inference.log 2>&1
echo "=== Evaluation started at $(date) ==="

actor_model_path=models/gemma-2b-tldr-cdrlhf/step-11999
reward_model_path=models/gemma-2b-tldr-rm/step_18572

# reward 모델 점수 계산
echo "--- Calculating reward model scores..."
python evaluation/reward_model_score.py \
    --dataset-path openai/summarize_from_feedback \
    --model-path $actor_model_path \
    --model-name gemma-2b-tldr-cdrlhf \
    --reward-model $reward_model_path \
    --gpus $GPU_ID \
    --batch-size 16

# diversity 계산을 위한 샘플 생성
echo "--- Generating samples for diversity evaluation..."

python evaluation/generate_samples.py \
    --dataset-path openai/summarize_from_feedback \
    --model-path ${actor_model_path} \
    --model-name gemma-2b-tldr-cdrlhf \
    --gpus $GPU_ID

echo "--- Calculating diversity metrics..."
python evaluation/diversity_eval.py \
    --file $basepath/results/generated/summarize_from_feedback/gemma-2b-tldr-cdrlhf.jsonl