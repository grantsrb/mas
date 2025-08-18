#!/bin/bash

# This script initiates a number of finetunings in different tmux
# windows within the tmux session from which it was run.

# Comma-separated list → array -- will use gpu pairs for each run
#cuda_devices="0,1,2,3,4,5,6,7,8,9"
cuda_devices="0,1,2,3,4,5"

split_idxs=(0 1 2)
filter_modes=("toxic" "nontoxic")

arg0="n_splits=3"
arg1="root_dir=./"
arg2="lr=0.0005"
arg3="model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
arg4=""
arg5=""
arg6=""
arg7=""
arg8=""

IFS=',' read -ra CUDA_LIST <<< "$cuda_devices"
# Get current tmux session name
SESSION_NAME=$(tmux display-message -p '#S')
if [ -z "$SESSION_NAME" ]; then
  echo "❌ This script must be run from inside a tmux session."
  exit 1
fi

echo "🚀 Dispatching jobs to tmux windows using 1 GPU per run..."

job_idx=0

for filter_mode in "${filter_modes[@]}"; do
    for split_idx in "${split_idxs[@]}"; do
        # Get the CUDA device
        gpu="${CUDA_LIST[$job_idx]}"
        window_name="ftune_${job_idx}_gpu${gpu}"

        # Build the command
        CMD="CUDA_VISIBLE_DEVICES=${gpu} python3 huggingface_finetuning.py \
            filter_mode=$filter_mode split_idx=$split_idx\
            $arg0 $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 \
            $1 $2 $3 $4 $5 $6 $7 $8; exec bash"

        echo "🧠 Launching $window_name with GPU $gpu"

        # Launch the command in a new tmux window
        tmux new-window -t "$SESSION_NAME" -n "$window_name" "bash -c '$CMD'"

        ((job_idx++))
        sleep 2
    done
done

