import wandb

def init_wandb(args, ds_config, project_name="gemma-sft", group_name=None, tags=None):
    """
        Rank 0에서만 W&B를 초기화합니다.
        group_name: "1-SFT", "2-RewardModel", "3-CD-RLHF" 등 큰 단계
        tags: ["gemma-2b"] 등 세부 키워드 리스트
    """
    if args.global_rank == 0:
        full_config = vars(args).copy()
        if ds_config is not None:
            full_config.update(ds_config) # Deepspeed config도 W&B에 저장
        wandb.init(
            project=project_name,
            group=group_name,
            tags=tags,
            job_type="train",
            name=f"{group_name}-{args.actor_model_name_or_path.split('/')[-1]}",
            config=full_config,
        )

def log_metrics_wandb(mode, step_type, total_steps, global_rank, metrics, epoch=None):
    """
    mode: 'train' 또는 'eval'
    step_type: 'sft' (SFT용), 'rm' (리워드 모델용), 'cd_rlhf' (CD-RLHF용)
    metrics: 기록할 지표들이 담긴 딕셔너리
    """
    if global_rank != 0:
        return

    prefix = f"{mode}/"
    log_data = {}

    # 1. 공통 지표 (Loss, LR)
    if "loss" in metrics:
        val = metrics["loss"]
        log_data[f"{prefix}loss"] = val.item() if hasattr(val, 'item') else val
    if "lr" in metrics:
        log_data[f"{prefix}learning_rate"] = metrics["lr"]
    if epoch is not None:
        log_data["epoch"] = epoch

    # 2. LLM (SFT) 전용 지표
    if step_type == "sft":
        if "perplexity" in metrics:
            log_data[f"{prefix}perplexity"] = metrics["perplexity"]

    # 3. Reward Model 전용 지표
    elif step_type == "rm":
        if "acc" in metrics:
            log_data[f"{prefix}accuracy"] = metrics["acc"]
        if "chosen_score" in metrics and "reject_score" in metrics:
            c = metrics["chosen_score"]
            r = metrics["reject_score"]
            log_data[f"{prefix}chosen_score"] = c
            log_data[f"{prefix}reject_score"] = r
            log_data[f"{prefix}margin"] = c - r

    # 4. CD-RLHF 전용 지표
    elif step_type == "cd_rlhf":
        for key in metrics:
            val = metrics[key]
            log_data[f"{prefix}{key}"] = val.item() if hasattr(val, 'item') else val
        

    # W&B 기록
    wandb.log(log_data, step=total_steps)

def finish_wandb(global_rank):
    """W&B 세션을 종료합니다."""
    if global_rank == 0:
        wandb.finish()