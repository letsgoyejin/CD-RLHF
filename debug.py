import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_sft_model():
    tokenizer = AutoTokenizer.from_pretrained("/home/soo/yejin/CD-RLHF/models/gemma-2b-tldr-sft")
    model = AutoModelForCausalLM.from_pretrained("/home/soo/yejin/CD-RLHF/models/gemma-2b-tldr-sft")

    prompt = "POST\nSubreddit: r/relationships\nHey all...\nTL;DR: "
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))

def avg_reward(filepath):
    with open(filepath, "r") as f:
        data = [json.loads(line) for line in f]
    
    rewards = [d["reward"] for d in data]
    avg_reward = sum(rewards) / len(rewards)
    print(f"Average reward: {avg_reward}")


if __name__ == "__main__":
    # test_sft_model()
    avg_reward("results/rewards/summarize_from_feedback/gemma-2b-tldr-cdrlhf.jsonl")