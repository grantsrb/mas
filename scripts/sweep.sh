#!/bin/bash
# Use this script to run the DAS experiments

gpus=( 0 1 2 3 4 5 6 7 8 9 )
exp_name="sweep"
model_path1="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_gru/multiobject_gru_0_seed12345"
model_path2="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_gru/multiobject_gru_1_seed23456"
config="configs/numequiv_indywise.yaml"
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
        cuda=${gpus[$cuda_idx]}
        echo s1 $s1
        echo s2 $s2
        echo m1 $model_path1
        echo m2 $model_path2

        bash scripts/single_model_run.sh $cuda $exp_name $config $model_path1 $model_path2 $s1 $s2 $arg1 $arg2 $arg3 $arg4 &
        echo DONE

        cuda_idx=$((1+$cuda_idx))
        if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
            cuda_idx=0
        fi
        sleep 0.75
    done
done






