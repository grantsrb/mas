#!/bin/bash

cuda_devices=0,1,2,3,5,7
max_samples=400

CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_generate_source_data.py\
    dataset="anitamaxvim/jigsaw-toxic-comments"\
    model_name="/data2/grantsrb/mas_finetunings/nontoxic_Anthropic-hh-rlhf_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t16-31-08" \
    max_samples=$max_samples
CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_generate_source_data.py \
    dataset="anitamaxvim/jigsaw-toxic-comments"\
    model_name="/data2/grantsrb/mas_finetunings/toxic_Anthropic-hh-rlhf_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t14-53-43" \
    max_samples=$max_samples

CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_generate_source_data.py\
    dataset="anitamaxvim/jigsaw-toxic-comments"\
    model_name="/data2/grantsrb/mas_finetunings/nontoxic_anitamaxvim-jigsaw-toxic-comments_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t13-16-04" \
    max_samples=$max_samples
CUDA_VISIBLE_DEVICES=$cuda_devices  python3 huggingface_generate_source_data.py \
    dataset="anitamaxvim/jigsaw-toxic-comments"\
    model_name="/data2/grantsrb/mas_finetunings/toxic_anitamaxvim-jigsaw-toxic-comments_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t11-40-08" \
    max_samples=$max_samples

CUDA_VISIBLE_DEVICES=$cuda_devices  python3 huggingface_generate_source_data.py\
    dataset="anitamaxvim/jigsaw-toxic-comments"\
    model_name="/data2/grantsrb/mas_finetunings/nontoxic_lmsys-toxic-chat_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t21-17-25" \
    max_samples=$max_samples
CUDA_VISIBLE_DEVICES=$cuda_devices  python3 huggingface_generate_source_data.py \
    dataset="anitamaxvim/jigsaw-toxic-comments"\
    model_name="/data2/grantsrb/mas_finetunings/toxic_lmsys-toxic-chat_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t18-08-00" \
    max_samples=$max_samples

