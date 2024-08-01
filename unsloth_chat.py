from unsloth import FastLanguageModel
import torch
import os
from typing import Dict, List, Any
from huggingface_hub import login
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

from datasets import load_dataset


MAX_SEQ_LEN = 128000
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True # Use 4bit quantization to reduce memory usage. Can be False.

# LoRA
R = 32
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]
LORA_ALPHA = 16
LORA_DROPOUT = 0.05 # Supports any, but = 0 is optimized
BIAS = "none" # Supports any, but = "none" is optimized
USE_GRADIENT_CHECKPOINTING = "unsloth"


# WANDB
WANDB_PROJECT = "flow-lm-judge-llama_31_8b_binary"
WANDB_ENTITY = "bergr7_"

os.environ["WANDB_PROJECT"] = WANDB_PROJECT
os.environ["WANDB_ENTITY"] = WANDB_ENTITY
# ! wrong -  fix paths that I mistakenly overwrote
def run():
    
    login()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length = MAX_SEQ_LEN,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        token = os.getenv("HF_TOKEN", None)
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=BIAS,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
    )
    
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    dataset = load_dataset("flowaicom/ft_e_evals_dataset_full-v0_binary")
    
    train_ds = dataset["train"].map(formatting_prompts_func, batched = True)
    validation_ds = dataset["validation"].map(formatting_prompts_func, batched = True)
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_ds,
        eval_dataset=validation_ds,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LEN,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            eval_strategy="steps",
            eval_steps=10,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs=1,
            learning_rate = 0.0002,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs/binary_flow-lm-judge-llama_31_8b_chat",
            report_to="wandb",
            run_name="three_point_first_run_batch1",
            save_strategy="steps",
            save_steps=250,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            push_to_hub=True,
            hub_strategy="end",
            hub_private_repo=True,
            push_to_hub_model_id="binary_flow-lm-judge-llama_31_8b_chat",
            push_to_hub_organization="flowaicom",
        ),
    )
    
    trainer.train()
    
    model.push_to_hub("flowaicom/binary_flow-lm-judge-llama_31_8b_chat_runpod", repo_type="model", private=True)
    tokenizer.push_to_hub("flowaicom/binary_flow-lm-judge-llama_31_8b_chat_runpod", repo_type="model", private=True)
    
    
if __name__ == "__main__":
    run()