#!/bin/bash
# Use this script to run individual experiments

# Uses command line arguments
cuda=$1
exp_name=$2
model_path1=$3
config=$4
IFS=' ' read -r -a search1 <<< "$5"
IFS=' ' read -r -a search2 <<< "$6"
arg3=$7
arg4=$8
arg5=$9

for arg1 in ${search1[@]}
do
    for arg2 in  ${search2[@]}
    do
        i=0
        while tmux has-session -t "${exp_name}${i}-${cuda}" 2>/dev/null; do
            ((i++))
        done
        session_name="${exp_name}${i}-${cuda}"
        
        tmux new -d -s "$session_name" "export CUDA_VISIBLE_DEVICES=${cuda}; export CUBLAS_WORKSPACE_CONFIG=:4096:8; python3 main.py $config model_names=$model_path1, $arg1 $arg2 $arg3 $arg4 $arg5 ; tmux wait -S ${session_name}"

        
        printf "\nCUDA$cuda M1: $model_path1 \nCUDA_VISIBLE_DEVICES=${cuda} python3 main.py $config model_names=$model_path1, $arg1 $arg2 $arg3 $arg4 $arg5 \n"
        echo "Waiting on ${session_name}"
        tmux wait ${session_name}
    done
done

