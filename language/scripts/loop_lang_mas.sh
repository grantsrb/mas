#!/bin/bash

# This script initiates a number of mas trainings in different tmux
# windows within the tmux session from which it was run.

# Comma-separated list → array -- will use gpu pairs for each run
#cuda_devices="0,1,2,3,4,5,6,7,8,9"
cuda_devices="0,1,2,3,4,5,6,7,8,9"

model_folders=(
    "/mnt/fs2/grantsrb/split_finetunes/nontoxic_alltoxic-0o3_EleutherAI-pythia-410m/"
    "/mnt/fs2/grantsrb/split_finetunes/nontoxic_alltoxic-1o3_EleutherAI-pythia-410m/"
    "/mnt/fs2/grantsrb/split_finetunes/nontoxic_alltoxic-2o3_EleutherAI-pythia-410m/"
    "/mnt/fs2/grantsrb/split_finetunes/toxic_alltoxic-0o3_EleutherAI-pythia-410m/"
    "/mnt/fs2/grantsrb/split_finetunes/toxic_alltoxic-1o3_EleutherAI-pythia-410m/"
    "/mnt/fs2/grantsrb/split_finetunes/toxic_alltoxic-2o3_EleutherAI-pythia-410m/"
)

arg1="n_train_samples=10000"
arg2="lr=0.002"
arg3=""
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

echo "🚀 Dispatching jobs to tmux windows using 2 GPUs per run..."

job_idx=0
gpu_pair_idx=0
num_devices=${#CUDA_LIST[@]}

for ((i = 0; i+1 < ${#model_folders[@]}; i++)); do # Full loop
    model_folder1="${model_folders[$i]}"

    # Create a list of windows to track
    launched_windows=()

    for ((j = i + 1; j < ${#model_folders[@]}; j++)); do
        model_folder2="${model_folders[$j]}"

        if (( gpu_pair_idx + 1 >= num_devices )); then
            echo "❗ Not enough GPUs left for a pair at index $gpu_pair_idx. Skipping."
            break
        fi

        gpu1="${CUDA_LIST[$gpu_pair_idx]}"
        gpu2="${CUDA_LIST[$((gpu_pair_idx + 1))]}"
        device_pair="$gpu1,$gpu2"
        window_name="job_${job_idx}_gpu${gpu1}_${gpu2}"

        CMD="CUDA_VISIBLE_DEVICES=${device_pair} python3 lang_mas.py \
            source_files=${model_folder1},${model_folder2} \
            $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 \
            $1 $2 $3 $4 $5 $6 $7 $8"

        echo "🧠 Launching $window_name with GPUs $device_pair"
        tmux new-window -t "$SESSION_NAME" -n "$window_name" "bash -c '$CMD'"

        echo sesh name: $SESSION_NAME

        launched_windows+=("$window_name")
        ((job_idx++))
        ((gpu_pair_idx+=2))
        sleep 2
    done

    # Function to check if a tmux window still exists
    tmux_window_exists() {
        tmux list-windows -t "$SESSION_NAME" 2>/dev/null | grep -q "$1"
    }
    
    # Wait loop: only wait for windows that are still alive
    echo "⏳ Waiting for jobs to finish for $model_folder1..."
    while true; do
        remaining=0
        for win in "${launched_windows[@]}"; do
            if tmux_window_exists "$win"; then
                ((remaining++))
            fi
        done
        if [ "$remaining" -eq 0 ]; then
            echo "✅ All jobs for $model_folder1 completed."
            break
        fi
        sleep 5
    done


    # Reset GPU index for next batch (optional if GPU usage is sequential)
    gpu_pair_idx=0
done
