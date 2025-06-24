#!/bin/bash/

cuda=9
layers="inpt_identity,inpt_identity"
root_folder="/mnt/fs2/grantsrb"
exp_folders=( "mas_neurips2025/multiobject_gru" "mas_moreseeds_neurips2025/multiobject_gru" "multiobject_rope_tformer_unk" )
#"multiobjectmod_gru"
#"multiobjectround_rope_tformer_unk"

all_exp_folders=""
for exp_folder in ${exp_folders[@]}
do
    all_exp_folders="${all_exp_folders} ${root_folder}/${exp_folder}"
done

echo python3 similarity.py $all_exp_folders layers=$layers
CUDA_VISIBLE_DEVICES=$cuda python3 similarity.py $all_exp_folders layers=$layers
