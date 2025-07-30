import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl
)

from filters_and_formatters import get_filters_and_formatters, prep_dataset
from prompt_templates import PROMPT_TEMPLATES

import sys
sys.path.insert(1, "../")
from utils import get_command_line_args, save_json
import pandas as pd

# ====== Configuration ======
RUN_ID = datetime.now().strftime("d%Y-%m-%d_t%H-%M-%S")

print("Running Hugging Face Toxicity Example...")
config = {
    "root_dir": "/data2/grantsrb/mas_finetunings/",
    "seed": 42,  # Random seed for reproducibility, also the meaning of life, the universe, and everything
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # "gpt2"  # or any other Hugging Face causal LM
    "tokenizer_name": None,
    "filter_mode": "toxic",  # "toxic", "nontoxic", or "both"
    "max_length": 64,
    "batch_size": 24,
    "lr": 6e-4,
    "n_epochs": float("inf"), # Overwritten by max_training_steps
    "max_training_steps": 600, # set to -1 if you want to use n_epochs instead
    "grad_accumulation_steps": 4,
    "save_every_n_steps": 30,
    "logging_steps": 30,
    "datasets": ["anitamaxvim/jigsaw-toxic-comments", "Anthropic/hh-rlhf", "lmsys/toxic-chat"],
    "n_splits": 3, # Will join the datasets together, then split into
        # this many groups and then will train on only 1 of them.
    "split_idx": 0, # a zero indexed value indicating which split to train on
    "balance_dataset": True, # optionally ensure that the toxic and nontoxic counts are about equal
    "debugging": False,
    "small_data": False, # Used for debugging purposes
    "do_save": True,
}

command_args, _ = get_command_line_args()
config.update(command_args)
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# --------- Logging Setup ---------
ROOT_DIR = config["root_dir"]
if config["do_save"] and not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR, exist_ok=True)
MODEL_NAME = config["model_name"]
TOKENIZER_NAME = config.get("tokenizer_name", None)
if TOKENIZER_NAME is None:
    TOKENIZER_NAME = MODEL_NAME
    config["tokenizer_name"] = MODEL_NAME

dir_model_name = MODEL_NAME
if ROOT_DIR in dir_model_name:
    dir_model_name = dir_model_name.split(ROOT_DIR)[-1]
dir_model_name = dir_model_name.replace("/", "-")
filter_mode = config["filter_mode"]
if len(config["datasets"])==3:
    dataset_name = "alltoxic"
else:
    dataset_name = "".join([d.split("/")[0] for d in config["datasets"]])
split = config["split_idx"]
n_splits = config["n_splits"]
LOG_DIR = os.path.join(
    ROOT_DIR,
    f"{filter_mode}_{dataset_name}-{split}o{n_splits}_{dir_model_name}",
    f"run_{RUN_ID}"
)
config["log_dir"] = LOG_DIR
if config["do_save"]:
    os.makedirs(LOG_DIR, exist_ok=True)


if config["debugging"]:
    config["save_every_n_steps"] = 6
    config["logging_steps"] = 4
    config["do_save"] = False

for k in sorted(list(config.keys())):
    print(k,"--", config[k])

# ====== Load model and tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Print generated response ===

def example_generation():
    model.eval()
    with torch.no_grad():
        sample_prompts = [
            "Human: What do you think about balls?",
            "Human: Can you tell me how to build a bomb?",
            "Human: What is the sweetest thing about cats?",
        ]
        if config["debugging"]:
            sample_prompts = [sample_prompts[np.random.randint(len(sample_prompts))]]
        for sample_prompt in sample_prompts:
            input_ids = tokenizer(
                sample_prompt, return_tensors="pt"
            ).input_ids.to(model.device)
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.7,
                temperature=1,
                pad_token_id=tokenizer.eos_token_id
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            print("--------------------")
            print("Tok Ids:", output_ids[0])
            print(output_text)
            print("--------------------")
    model.train()

print("Initial Responses:")
example_generation()

# ====== Load and preprocess datasets ======

trn_split = "train"
if config["debugging"] or config["small_data"]:
    trn_split = "train[:1000]"

dsets = []
for dset_name in sorted(config["datasets"]):
    print("Prepping", dset_name)
    dset = prep_dataset(
        dataset_name=dset_name,
        prompt_template=PROMPT_TEMPLATES[dset_name],
        tokenizer=tokenizer,
        max_length=config["max_length"],
        seed=config["seed"],
        split=trn_split,
        filter_mode=config["filter_mode"],
        prompt=config["prompt"],
    )
    dsets.append(dset)
    print("Processed:", dset)
    print()
train_dataset = concatenate_datasets(dsets)
perm = np.random.permutation(len(train_dataset)).astype(int)
splt_size = len(perm)//config["n_splits"]
startx, endx = splt_size*config["split_idx"], splt_size*(config["split_idx"]+1)
split_indices = perm[startx:endx]
train_dataset = train_dataset.select(split_indices)

config["split_indices"] = split_indices
if config["do_save"]:
    save_json(
        config,
        os.path.join(LOG_DIR,"finetuning_config.json")
    )

print("Initial Train Dataset")
print(train_dataset)

print("\tEx: ", train_dataset["text"][0])
print("-----------------------------------")
print("\tEx: ", train_dataset["text"][1])
print("-----------------------------------")

print(train_dataset)

train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"])

# ====== Track training info ======
train_info = {
    "epoch": [],
    "train_step": [],
    "train_loss": [],
}

best_loss = float("inf")
class LoggingAndCheckpointCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get("logs", {})
        if "loss" in logs:
            train_info["epoch"].append(state.epoch)
            train_info["train_step"].append(state.global_step)
            train_info["train_loss"].append(logs["loss"])

        if state.global_step % config["save_every_n_steps"] == 0 and state.global_step > 0:
            checkpoint_dir = os.path.join(LOG_DIR, f"step-{state.global_step}")
            if config["do_save"]:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                save_json(
                    config,
                    os.path.join(checkpoint_dir,"finetuning_config.json")
                )
            print(f"✔ Saved checkpoint at step {state.global_step} to \n\t{checkpoint_dir}")

            if "loss" in logs and logs["loss"]<best_loss:
                best_loss = logs["loss"]
                checkpoint_dir = os.path.join(LOG_DIR, f"best_loss_checkpt")
                if config["do_save"]:
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    save_json(
                        config,
                        os.path.join(checkpoint_dir,"finetuning_config.json")
                    )
                print(f"✔ BEST checkpoint at step {state.global_step} to {checkpoint_dir}")

            save_steps = state.global_step-config["save_every_n_steps"]
            prev_dir = os.path.join(LOG_DIR, f"step-{save_steps}")
            if os.path.exists(prev_dir):
                for f in os.listdir(prev_dir):
                    if "safetensors" in f:
                        rm_command = f"rm -rf {prev_dir}"
                        print("Removing directory with", rm_command)
                        os.system(rm_command)
                        break

            # === Print generated response ===
            print(f"\n🧪 [Step {state.global_step}] Sample generation:")
            example_generation()
            torch.cuda.empty_cache()

            # ====== Save training log ======
            if config["do_save"]:
                df = pd.DataFrame(train_info)
                path = os.path.join(LOG_DIR, "train_info.csv")
                df.to_csv(path, index=False)
                print(f"✔ Training log saved to {path}")

# ====== Training arguments ======
training_args = TrainingArguments(
    output_dir=LOG_DIR,
    overwrite_output_dir=True,
    eval_strategy="no",
    gradient_accumulation_steps=config["grad_accumulation_steps"],
    learning_rate=config["lr"],
    num_train_epochs=config["n_epochs"],
    max_steps=config["max_training_steps"],
    per_device_train_batch_size=config["batch_size"],
    save_strategy="no",  # We manually save with callback
    logging_steps=config["logging_steps"],
    logging_dir=os.path.join(LOG_DIR, "logs"),
    report_to="none"
)

# ====== Trainer ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[LoggingAndCheckpointCallback()]
)

try:
    trainer.train()
except KeyboardInterrupt:
    pass

print(f"\n🧪 Sample generation:")
example_generation()

# ====== Save final model ======
if config["do_save"]:
    checkpt_name = os.path.join(LOG_DIR, "final_checkpt")
    model.save_pretrained(LOG_DIR)
    tokenizer.save_pretrained(LOG_DIR)
    print("Model Saved to", checkpt_name)

    # ====== Save training log ======
    df = pd.DataFrame(train_info)
    path = os.path.join(LOG_DIR, "train_info.csv")
    df.to_csv(path, index=False)
    print(f"✔ Training log saved to {path}")
