#!/bin/bash

cuda_devices="0,1,2,3,4,5,6,7"
max_samples=1000
temperature=0
top_p=1

model_folders=(
    "/data/grantsrb/split_finetunes/nontoxic_alltoxic-0o3_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/"
    "/data/grantsrb/split_finetunes/toxic_alltoxic-0o3_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/"
    "/data/grantsrb/split_finetunes/nontoxic_alltoxic-1o3_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/"
    "/data/grantsrb/split_finetunes/nontoxic_alltoxic-2o3_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/"
    "/data/grantsrb/split_finetunes/toxic_alltoxic-1o3_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/"
    "/data/grantsrb/split_finetunes/toxic_alltoxic-2o3_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/"
)

IFS=',' read -ra CUDA_LIST <<< "$cuda_devices"
# Get current tmux session name
SESSION_NAME=$(tmux display-message -p '#S')
if [ -z "$SESSION_NAME" ]; then
  echo "❌ This script must be run from inside a tmux session."
  exit 1
fi

echo "🚀 Dispatching jobs to tmux windows using 2 GPUs per run..."

job_idx=0
for model in ${model_folders[@]}
do
        # Get the CUDA device
        gpu="${CUDA_LIST[$job_idx]}"
        window_name="ftune_${job_idx}_gpu${gpu}"

        # Build the command
        CMD="CUDA_VISIBLE_DEVICES=${gpu} python3 huggingface_collect_actvs.py\
            model_name="${model}"\
            max_samples=$max_samples\
            temperature=$temperature\
            top_p=$top_p\
            $arg0 $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 \
            $1 $2 $3 $4 $5 $6 $7 $8; exec bash"

        echo "🧠 Launching $window_name with GPU $gpu"

        # Launch the command in a new tmux window
        tmux new-window -t "$SESSION_NAME" -n "$window_name" "bash -c '$CMD'"

        ((job_idx++))
        sleep 2
done

