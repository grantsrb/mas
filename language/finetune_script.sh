#!/bin/bash

#cuda_devices=0,1,2,3
#cuda_devices=4,5,6
#cuda_devices=7,8,9
cuda_devices=2
root_dir="/mnt/fs2/grantsrb/deep_finetunes/"
model_name="gpt2"
max_length=128
max_training_steps=500
lr=0.005

for dataset in "Anthropic/hh-rlhf" #"anitamaxvim/jigsaw-toxic-comments" #"lmsys/toxic-chat" #
do
    for filter_mode in "toxic" "nontoxic"
    do
        echo
        echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_finetuning.py\
            root_dir=$root_dir\
            dataset=$dataset\
            filter_mode=$filter_mode\
            model_name=$model_name\
            max_training_steps=$max_training_steps\
            lr=$lr\
            max_length=$max_length
        echo
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_finetuning.py\
            root_dir=$root_dir\
            dataset=$dataset\
            filter_mode=$filter_mode\
            model_name=$model_name\
            max_training_steps=$max_training_steps\
            lr=$lr\
            max_length=$max_length
    done
done
