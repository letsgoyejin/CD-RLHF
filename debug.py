from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/home/soo/yejin/CD-RLHF/models/gemma-2b-tldr-sft")
model = AutoModelForCausalLM.from_pretrained("/home/soo/yejin/CD-RLHF/models/gemma-2b-tldr-sft")

prompt = "POST\nSubreddit: r/relationships\nHey all...\nTL;DR: "
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))