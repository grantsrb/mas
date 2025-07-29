#!/bin/bash

cuda_devices=8
max_samples=1000
dataset="anitamaxvim/jigsaw-toxic-comments"

models=(
    #"/mnt/fs2/grantsrb/deep_finetunes/nontoxic_anitamaxvim-jigsaw-toxic-comments_gpt2"
    #"/mnt/fs2/grantsrb/deep_finetunes/nontoxic_lmsys-toxic-chat_gpt2"
    #"/mnt/fs2/grantsrb/deep_finetunes/toxic_lmsys-toxic-chat_gpt2"
    #"/mnt/fs2/grantsrb/deep_finetunes/toxic_anitamaxvim-jigsaw-toxic-comments_gpt2"
    "/mnt/fs2/grantsrb/deep_finetunes/nontoxic_Anthropic-hh-rlhf_gpt2/"
    "/mnt/fs2/grantsrb/deep_finetunes/toxic_Anthropic-hh-rlhf_gpt2/"
)

for model in ${models[@]}
do
    echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
        dataset=$dataset\
        max_samples=$max_samples\
        model_name="${model}"\
        temperature=0\
        top_p=1\
        $1 $2 $3 $4
    CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
        dataset=$dataset\
        model_name="${model}"\
        max_samples=$max_samples\
        temperature=0\
        top_p=1\
        $1 $2 $3 $4
done

