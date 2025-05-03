#!/bin/bash
# Use this script to run the DAS experiments

gpus=( 5 6 7 8 9 0 1 2 3 4 )
#gpus=( 0 1 2 3 4 5 6 7 8 9 )
exp_name="sweep"
model_path1="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_gru/multiobject_gru_0_seed12345"
#model_path2="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_lstm/multiobject_lstm_1_seed12345"
model_path2="/mnt/fs2/grantsrb/mas_neurips2025/sameobject_gru/sameobject_gru_1_seed12345"
#model_path2="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_rope_tformer_unk/multiobject_rope_tformer_unk_0_seed23456"

config="configs/numequiv_stepwise.yaml"
search1=("n_units=16" "n_units=4" "n_units=32" "n_units=64" "n_units=96" )
search2=("swap_keys=full" ) # "swap_keys=count")
arg1=""
arg2=""
arg3=""
arg4=""
arg5=""


echo Dispatching
cuda_idx=0
for s1 in  ${search1[@]}
do
    for s2 in  ${search2[@]}
    do
        echo ------------------
        cuda=${gpus[$cuda_idx]}
        echo search term1 $s1
        echo search term2 $s2
        echo model path1 $model_path1
        echo model path2 $model_path2

        bash scripts/single_model_run.sh $cuda $exp_name $config $model_path1 $model_path2 $s1 $s2 $arg1 $arg2 $arg3 $arg4 &

        cuda_idx=$((1+$cuda_idx))
        if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
            cuda_idx=0
        fi
        sleep 0.75
    done
done






