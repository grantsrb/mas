#!/bin/bash
# Use this script to run the DAS experiments

exp_name="mas_dispatch"
#gpus=( 0 1 2 3 4 5 6 7 8 9 )
gpus=( 6 7 8 9 )
root_folder="/mnt/fs2/grantsrb/mas_saves/"
exp_folders1=( "big-multiobj"  ) #"big-multiobj" "lstm_multiobj" "tformer_multiobj_unk_d40" "big-sameobj" "lstm_sameobj" )
exp_folders2=( "big-multiobj" )
config="configs/numequiv.yaml"
n_units="n_units=24"
fsr="fsr=False"
consistent_targ_inpt_id="consistent_targ_inpt_id=False"

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

                           bash scripts/single_model_run.sh $cuda $exp_name $model_path1 $model_path2 $config $n_units $fsr $consistent_targ_inpt_id &

                           cuda_idx=$((1+$cuda_idx))
                           if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
                               cuda_idx=0
                           fi
                      fi
                done
            done
         fi
    done
done
