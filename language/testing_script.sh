#!/bin/bash/
#Use this script to ensure that the lang_mas.py script is doing what it
#should be.

cuda_devices=0,1
max_samples=400
model="waghmareps12/SmolLM_125M"
dataset="anitamaxvim/jigsaw-toxic-comments"
root_dir="/data2/grantsrb/mas_finetunings/"


# First we need to make the activations
echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
    dataset=$dataset\
    max_samples=$max_samples\
    temperature=0\
    top_p=1\
    model_name="${root_dir}${model}"\
    $1 $2 $3 $4
CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
    dataset=$dataset\
    temperature=0\
    top_p=1\
    max_samples=$max_samples\
    model_name="${root_dir}${model}"\
    $1 $2 $3 $4


# Next we need to try running mas on the generated activations to ensure
# that a mapping can be learned

echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
    n_train_samples=500\
    n_valid_samples=100\
    n_units=100000\
    identity_rot=True\
    source_files="${root_dir}${model},${root_dir}${model}"
CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
    n_train_samples=500\
    n_valid_samples=100\
    n_units=100000\
    identity_rot=True\
    source_files="${root_dir}${model},${root_dir}${model}"
