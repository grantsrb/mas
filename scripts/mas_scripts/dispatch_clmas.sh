#!/bin/bash
# Use this script to run the DAS experiments

exp_name="clvecs"
#gpus=( 0 1 2 3 4 5 6 7 8 9 )
gpus=( 4 5 6 7 8 9 )
root_folder="/mnt/fs2/grantsrb/mas_neurips2025/"


exp_folders1=( "multiobject_gru" ) # "sameobject_gru" "multiobject_lstm" ) # "multiobject_rope_tformer_unk" "sameobject_lstm" 
exp_folders2=( "multiobject_gru" )
config="configs/cl_mas.yaml"
search1=( "n_units=64" "n_units=128" )
search2=( "cl_eps=8" "cl_eps=13" ) # "swap_keys=count" ) 
arg1="swap_keys=full"
arg2=""

echo Dispatching
cuda_idx=0
for exp_folder1 in ${exp_folders1[@]}
do
    exp_root1="${root_folder}${exp_folder1}"
    for model_folder1 in `ls ${exp_root1}`; do
         if [[ $model_folder1 != *results* && $model_folder1 != *.json* ]]; then
            for exp_folder2 in ${exp_folders2[@]}
            do
                exp_root2="${root_folder}${exp_folder2}"
                for model_folder2 in `ls ${exp_root2}`; do
                     if [[ $model_folder2 != *results* && $model_folder2 != $model_folder1 && $model_folder2 != *.json* ]]; then

                           cuda=${gpus[$cuda_idx]}

                           model_path1="${exp_root1}/${model_folder1}"
                           model_path2="${exp_root2}/${model_folder2}"
                           echo Search1 ${search1[@]}
                           echo Search2 ${search2[@]}

                           bash scripts/mas_scripts/single_model_search.sh $cuda $exp_name $model_path1 $model_path2 $config "${search1[*]}" "${search2[*]}" $arg1 $arg2 &

                           cuda_idx=$((1+$cuda_idx))
                           if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
                               cuda_idx=0
                           fi
                           sleep 0.7
                      fi
                done
            done
         fi
    done
done
