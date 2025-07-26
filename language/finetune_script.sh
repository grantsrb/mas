#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python3 huggingface_finetuning.py filter_mode="toxic"    dataset="anitamaxvim/jigsaw-toxic-comments"
CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python3 huggingface_finetuning.py filter_mode="nontoxic" dataset="anitamaxvim/jigsaw-toxic-comments"

CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python3 huggingface_finetuning.py filter_mode="toxic"    dataset="Anthropic/hh-rlhf"
CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python3 huggingface_finetuning.py filter_mode="nontoxic" dataset="Anthropic/hh-rlhf"

CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python3 huggingface_finetuning.py filter_mode="toxic"    dataset="lmsys/toxic-chat"
CUDA_VISIBLE_DEVICES=0,1,2,3,5,7 python3 huggingface_finetuning.py filter_mode="nontoxic" dataset="lmsys/toxic-chat"
