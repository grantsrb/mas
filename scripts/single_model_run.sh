#!/bin/bash
# Use this script to run individual experiments

# Uses command line arguments
cuda=$1
exp_name=$2
config=$3
model_path1=$4
model_path2=$5

procs=()
i=0
while tmux has-session -t "${exp_name}${i}-${cuda}" 2>/dev/null; do
    ((i++))
done
session_name="${exp_name}${i}-${cuda}"

echo tmux new -d -s "$session_name" "export CUDA_VISIBLE_DEVICES=${cuda}; export CUBLAS_WORKSPACE_CONFIG=:4096:8; python3 main.py $config model_names=${model_path1},${model_path2} $5 $6 $7 $8 $9 ${10}; tmux wait -S ${session_name}"

printf "\nCUDA_VISIBLE_DEVICES=${cuda} python3 run_das.py $config model_names=${model_path1},${model_path2} $5 $6 $7 $8 $9 ${10}\n"
echo "Waiting on ${session_name}"
tmux wait ${session_name}



