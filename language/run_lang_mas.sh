#!/bin/bash

# This script initiates a number of mas trainings in different tmux
# windows within the tmux session from which it was run.

# Comma-separated list → array -- will use gpu pairs for each run
cuda_devices="0,1,2,3,4,5,6,7,8,9"

model_folder1="/mnt/fs2/grantsrb/mas_finetunings/nontoxic_anitamaxvim-jigsaw-toxic-comments_EleutherAI-pythia-410m"
model_folders=(
    "/mnt/fs2/grantsrb/mas_finetunings/nontoxic_Anthropic-hh-rlhf_EleutherAI-pythia-410m"
    "/mnt/fs2/grantsrb/mas_finetunings/toxic_Anthropic-hh-rlhf_EleutherAI-pythia-410m"

    "/mnt/fs2/grantsrb/mas_finetunings/nontoxic_lmsys-toxic-chat_EleutherAI-pythia-410m"
    "/mnt/fs2/grantsrb/mas_finetunings/toxic_lmsys-toxic-chat_EleutherAI-pythia-410m"

    "/mnt/fs2/grantsrb/mas_finetunings/nontoxic_anitamaxvim-jigsaw-toxic-comments_EleutherAI-pythia-410m"
    "/mnt/fs2/grantsrb/mas_finetunings/toxic_anitamaxvim-jigsaw-toxic-comments_EleutherAI-pythia-410m"
)


dataset="anitamaxvim/jigsaw-toxic-comments"
arg1="n_train_samples=10000"
arg2="lr=0.005"
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

for model_folder2 in "${model_folders[@]}"; do
  # Check if there are at least two GPUs left
  if (( gpu_pair_idx + 1 >= num_devices )); then
    echo "❗ Not enough GPUs left for a pair at index $gpu_pair_idx. Skipping."
    break
  fi

  # Get the pair of CUDA devices
  gpu1="${CUDA_LIST[$gpu_pair_idx]}"
  gpu2="${CUDA_LIST[$((gpu_pair_idx + 1))]}"
  device_pair="$gpu1,$gpu2"
  window_name="job_${job_idx}_gpu${gpu1}_${gpu2}"

  # Build the command
  CMD="CUDA_VISIBLE_DEVICES=${device_pair} python3 lang_mas.py \
      source_files=${model_folder1},${model_folder2} \
      $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $arg7 $arg8 \
      $1 $2 $3 $4 $5 $6 $7 $8; exec bash"

  echo "🧠 Launching $window_name with GPUs $device_pair"

  # Launch the command in a new tmux window
  tmux new-window -t "$SESSION_NAME" -n "$window_name" "bash -c '$CMD'"

  ((job_idx++))
  ((gpu_pair_idx+=2))
done

