cd CD-RLHF
python evaluation/reward_model_score.py \
    --dataset-path openai/summarize_from_feedback \
    --model-path ${actor_model_path} \
    --model-name gemma-2b-tldr-rlhf \ 
    --reward-model ${reward_model_path} \
    --gpus 0,1,2,3,4,5,6,7 \
    --batch-size 16