#!/bin/bash
# Use this script to run the DAS experiments

exp_name="das"
gpus=( 0 1 2 3 4 5 6 7 8 9 )
root_folder="/mnt/fs2/grantsrb/mas_neurips2025/"

#"multiobjectmod_gru"
#"multiobjectround_gru"
#"multiobjectmod_lstm"
#"multiobjectround_lstm"



exp_folders1=(  "multiobjectmod_gru" ) #"multiobject_gru" ) # "sameobject_gru" "multiobject_lstm" "sameobject_lstm" "multiobjectmod_gru" )
config="configs/general_indywise.yaml"
search1=( "n_units=64" "n_units=96" "n_units=128" )
search2=( "swap_keys=full" )
arg1="mtx_types=RotationMatrix"
arg2=""
arg3=""

echo Dispatching
cuda_idx=0
for exp_folder1 in ${exp_folders1[@]}
do
    exp_root1="${root_folder}${exp_folder1}"
    for model_folder1 in `ls ${exp_root1}`; do
         if [[ $model_folder1 != *results* && $model_folder1 != *.json* ]]; then
           cuda=${gpus[$cuda_idx]}

           model_path1="${exp_root1}/${model_folder1}"
           model_path2="${exp_root2}/${model_folder2}"
           echo out1 ${search1[@]}
           echo out2 ${search2[@]}

           bash scripts/das_scripts/single_model_search.sh $cuda $exp_name $model_path1 $config "${search1[*]}" "${search2[*]}" $arg1 $arg2 $arg3 &

           cuda_idx=$((1+$cuda_idx))
           if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
               cuda_idx=0
           fi
           sleep 0.6
         fi
    done
done
