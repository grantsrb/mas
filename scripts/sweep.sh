#!/bin/bash
# Use this script to run the DAS experiments

gpus=( 0 1 2 3 4 5 6 7 8 9 )
exp_name="sweep"
model_path1="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_rope_tformer_unk/multiobject_rope_tformer_unk_0_seed23456/"
model_path2="/mnt/fs2/grantsrb/mas_neurips2025/multiobject_rope_tformer_unk/multiobject_rope_tformer_unk_0_seed23456/"
#model_path2=" "

config="configs/general_stepwise.yaml"
search1=( "n_units=16" "n_units=32" "n_units=48"  "n_units=64" )
search2=( "lr=0.0005" "lr=0.001" "lr=0.005" ) 
arg1="swap_keys=full"
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






