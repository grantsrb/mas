#!/bin/bash
# Use this script to run the DAS experiments

exp_name="upup"
gpus=( 0 1 2 3 4 5 6 7 8 9 )
root_folder="./models/"

exp_folders1=( "multiobjectmod_gru" ) #"multiobject_gru" ) # "sameobject_gru" "multiobject_lstm" "sameobject_lstm" "multiobjectmod_gru" )
configs=("configs/orthogonal_das.yaml" "configs/linear_das.yaml")
search1=( "n_units=1" "n_units=2" "n_units=4" "n_units=16" "n_units=32" )
search2=( "swap_keys=demo_count" "swap_keys=resp_count" "swap_keys=phase" )
arg1="cmodel_names=CountUpUp"
arg2=""
arg3=""

echo Dispatching
cuda_idx=0
for exp_folder1 in ${exp_folders1[@]}
do
    exp_root1="${root_folder}${exp_folder1}"
    for model_folder1 in `ls ${exp_root1}`; do
        if [[ $model_folder1 != *results* && $model_folder1 != *.json* ]]; then
            for config in ${configs[@]}; do
                 cuda=${gpus[$cuda_idx]}

                 model_path1="${exp_root1}/${model_folder1}"
                 model_path2="${exp_root2}/${model_folder2}"
                 echo Search1 ${search1[@]}
                 echo Search2 ${search2[@]}

                 bash scripts/das_scripts/single_model_search.sh $cuda $exp_name $model_path1 $config "${search1[*]}" "${search2[*]}" $arg1 $arg2 $arg3 &

                 cuda_idx=$((1+$cuda_idx))
                 if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
                     cuda_idx=0
                 fi
                 echo -----------------
                 sleep 0.7
            done
        fi
    done
done
