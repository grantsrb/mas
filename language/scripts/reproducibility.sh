#!/bin/bash/
#Use this script to finetune, collect activations, and then perform
# das to determine the number of dims needed for mas

cuda_devices=0,1
model="gpt2"
finetune_dataset="lmsys/toxic-chat"
filter_mode="toxic" # "toxic" or "nontoxic" or "both"
actvs_dataset="anitamaxvim/jigsaw-toxic-comments"
root_dir="./"

CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_finetuning.py\
    dataset=$finetune_dataset\
    filter_mode=$filter_mode\
    model_name=$model\
    max_training_steps=300\
    max_length=128


max_samples=1000

# First we need to make the activations
echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
    dataset=$dataset\
    max_samples=$max_samples\
    temperature=0\
    top_p=1\
    model_name="${root_dir}${model}"\
    $1 $2 $3 $4 $5 $6 $7 $8
CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
    dataset=$dataset\
    temperature=0\
    top_p=1\
    max_samples=$max_samples\
    model_name="${root_dir}${model}"\
    $1 $2 $3 $4 $5 $6 $7 $8




# Next we need to try running mas on the generated activations to ensure
# that a mapping can be learned

echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
    n_train_samples=500\
    n_valid_samples=100\
    n_units=100000\
    source_files="${root_dir}${model},${root_dir}${model}"\
    $1 $2 $3 $4 $5 $6 $7 $8
CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
    n_train_samples=500\
    n_valid_samples=100\
    n_units=100000\
    source_files="${root_dir}${model},${root_dir}${model}"\
    $1 $2 $3 $4 $5 $6 $7 $8
