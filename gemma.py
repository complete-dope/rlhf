# starting the gemma model 

import transformers 
# import trl 
# import torch 
# import torch.nn as nn 
import os 
from datasets import load_dataset

FILE_PATH = os.path.dirname(__file__)

def load_model():
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", cache_dir = FILE_PATH)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m-it", cache_dir = FILE_PATH)
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

# now lets do rlhf on the above model 

def load_dataset():
    ds = load_dataset("Dahoas/rm-static" ,cache_dir = FILE_PATH)      # train/test splits included
    # entries have: prompt, chosen, rejected (and response fields)
    print(len(ds["train"]))                    # ~76k
    example = ds["train"][0]
    print(example["prompt"])
    print("\n----\nChoosen answer is : ", example["chosen"][:200])

# Dahoas/rm-static