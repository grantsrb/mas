#!/bin/bash
# Use this script to run the DAS experiments

#gpus=( 0 1 2 3 4 5 6 7 8 9 )
gpus=( 9 1 2 ) #6 7 0 3 4 5 8 
exp_name="sweep"
model_path1="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_gru/multiobject_gru_1_seed23456/"
model_path2="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_gru/multiobject_gru_0_seed12345/"
#model_path2=" "

#config="configs/model_stitching.yaml"
#search1=( "n_units=32" )
config="configs/unimas.yaml"
search1=( "n_units=2" "n_units=4" "n_units=16" )
search2=( "swap_keys=full" ) 
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

        bash scripts/single_model_run.sh $cuda $exp_name $config $model_path1 "${model_path2}" $s1 $s2 $arg1 $arg2 $arg3 $arg4 &

        cuda_idx=$((1+$cuda_idx))
        if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
            cuda_idx=0
        fi
        sleep 0.75
    done
done






