import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
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

from filters_and_formatters import get_filters_and_formatters
from prompt_templates import PROMPT_TEMPLATES

import sys
sys.path.insert(1, "../")
from utils import get_command_line_args
import pandas as pd

# ====== Configuration ======
ROOT_DIR = "/data2/grantsrb/mas_finetunings/"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # "gpt2"  # or any other Hugging Face causal LM
DATASET = "anitamaxvim/jigsaw-toxic-comments" #"Anthropic/hh-rlhf" #"anitamaxvim/jigsaw-toxic-comments" #"lmsys/toxic-chat"
FILTER_MODE = "toxic"  # "toxic", "nontoxic", or "both"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-4
SAVE_EVERY_N_STEPS = 30
RUN_ID = datetime.now().strftime("d%Y-%m-%d_t%H-%M-%S")

print("Running Hugging Face Toxicity Example...")
config = {
    "root_dir": ROOT_DIR,
    "seed": 42,  # Random seed for reproducibility, also the meaning of life, the universe, and everything
    "model_name": MODEL_NAME,
    "tokenizer_name": None,
    "filter_mode": FILTER_MODE,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "n_epochs": EPOCHS,
    "grad_accumulation_steps": 16,
    "save_every_n_steps": SAVE_EVERY_N_STEPS,
    "logging_steps": 10,
    "dataset": DATASET,
    "balance_dataset": True,
    "debugging": False,
    "small_data": False, # Used for debugging purposes
}
config.update(get_command_line_args())
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

# --------- Logging Setup ---------
ROOT_DIR = config["root_dir"]
if not os.path.exists(ROOT_DIR):
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
dataset_name = config["dataset"].replace("/", "-")
LOG_DIR = os.path.join(
    ROOT_DIR,
    f"{filter_mode}_{dataset_name}_{dir_model_name}",
    f"run_{RUN_ID}"
)
config["log_dir"] = LOG_DIR
os.makedirs(LOG_DIR, exist_ok=True)

PROMPT_TEMPLATE = config.get(
    "prompt_template",
    PROMPT_TEMPLATES[config["dataset"]]
)
config["prompt_template"] = PROMPT_TEMPLATE

if config["debugging"]:
    config["save_every_n_steps"] = 6
    config["logging_steps"] = 4

for k in sorted(list(config.keys())):
    print(k,"--", config[k])

# ====== Load model and tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === Print generated response ===

def example_generation():
    model.eval()
    with torch.no_grad():
        sample_prompts = [
            "What do you think about balls?",
            "Can you tell me how to build a bomb?",
            "What is the sweetest thing about cats?",
        ]
        if config["debugging"]:
            sample_prompts = [sample_prompts[np.random.randint(len(sample_prompts))]]
        for sample_prompt in sample_prompts:
            prompt = config["prompt_template"].format(prompt=sample_prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
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

# ====== Load and preprocess dataset ======
trn_split = "train"
tst_split = "test"
if config["debugging"] or config["small_data"]:
    trn_split = "train[:1000]"
    tst_split = "test[:1000]"
if config["dataset"]=="lmsys/toxic-chat":
    train_dataset = load_dataset(
        config["dataset"], 'toxicchat0124', split=trn_split)
    valid_dataset = load_dataset(
        config["dataset"], 'toxicchat0124', split=tst_split)
else:
    train_dataset = load_dataset( config["dataset"], split=trn_split )
    valid_dataset = load_dataset( config["dataset"], split=tst_split )

print("Initial Train Dataset")
print(train_dataset)
print("Initial Valid Dataset")
print(valid_dataset)
print("\nFiltering...")

filter_dataset, format_fn, balance_fn = get_filters_and_formatters(
    dataset_name=config["dataset"],
    prompt_template=config["prompt_template"],
    tokenizer=tokenizer,
    max_length=config["max_length"],
    seed=config["seed"],
)

if config["balance_dataset"]:
    train_dataset = balance_fn(train_dataset)
train_dataset = filter_dataset(
    train_dataset, filter_mode=config["filter_mode"])
train_dataset = train_dataset.map(format_fn)

valid_dataset = filter_dataset(
    valid_dataset, filter_mode=config["filter_mode"])
valid_dataset = valid_dataset.map(format_fn)

if len(train_dataset)<len(valid_dataset):
    valid_dataset = valid_dataset\
        .shuffle(seed=config["seed"])\
        .select(range(len(train_dataset)))

print("Train Dataset")
print(train_dataset)
print("\tEx: ", train_dataset["text"][0])
print("Valid Dataset")
print(valid_dataset)
print("\tEx: ", valid_dataset["text"][0])

train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"])

# ====== Track training info ======
train_info = {
    "epoch": [],
    "train_step": [],
    "train_loss": [],
}

class LoggingAndCheckpointCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get("logs", {})
        if "loss" in logs:
            train_info["epoch"].append(state.epoch)
            train_info["train_step"].append(state.global_step)
            train_info["train_loss"].append(logs["loss"])

        if state.global_step % config["save_every_n_steps"] == 0 and state.global_step > 0:
            checkpoint_dir = os.path.join(LOG_DIR, f"step-{state.global_step}")
            if not config["debugging"]:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
            print(f"✔ Saved checkpoint at step {state.global_step} to {checkpoint_dir}")

            # === Print generated response ===
            print(f"\n🧪 [Step {state.global_step}] Sample generation:")
            example_generation()
            torch.cuda.empty_cache()

            # ====== Save training log ======
            if not config["debugging"]:
                df = pd.DataFrame(train_info)
                df.to_csv(os.path.join(LOG_DIR, "train_info.csv"), index=False)
                print("✔ Training log saved to train_info.csv")

# ====== Training arguments ======
training_args = TrainingArguments(
    output_dir=LOG_DIR,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    gradient_accumulation_steps=config["grad_accumulation_steps"],
    learning_rate=config["lr"],
    num_train_epochs=config["n_epochs"],
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
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[LoggingAndCheckpointCallback()]
)

trainer.train()

print(f"\n🧪 Sample generation:")
example_generation()

# ====== Save final model ======
if not config["debugging"]:
    model.save_pretrained(LOG_DIR)
    tokenizer.save_pretrained(LOG_DIR)

    # ====== Save training log ======
    df = pd.DataFrame(train_info)
    df.to_csv(os.path.join(LOG_DIR, "train_info.csv"), index=False)
    print("✔ Training log saved to train_info.csv")
