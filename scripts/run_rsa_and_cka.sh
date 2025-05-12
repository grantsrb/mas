#!/bin/bash/

layers="identities.0,identities.0"
root_folder="/mnt/fs2/grantsrb/mas_neurips2025"
exp_folders=( "multiobject_gru" "sameobject_gru" "multiobject_lstm" "sameobject_lstm" "multiobject_rope_tformer_unk" )
#"multiobjectmod_gru"
#"multiobjectround_rope_tformer_unk"

all_exp_folders=""
for exp_folder in ${exp_folders[@]}
do
    all_exp_folders="${all_exp_folders} ${root_folder}/${exp_folder}"
done

echo python3 similarity.py $all_exp_folders layers=$layers
python3 similarity.py $all_exp_folders layers=$layers
