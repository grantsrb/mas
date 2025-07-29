#!/bin/bash/
#Use this script to ensure that the lang_mas.py script is doing what it
#should be. Can argue `ignore` when running script to avoid recomputing
#source activations.
#Some parameters to play with are the following:
#    debugging=True
#    shuffle_intrv_data=False
#    max_length=20
#    layers=transformers.wte
#    identity_rot=True
#    n_units=10000
#    swap_keys="full" or "null_varb"
#
#Remember that you must use the embedding layer to completely transfer
#behavior between transformers.

cuda_devices=0,1
max_samples=25
model="toxic_lmsys-toxic-chat_gpt2/run_d2025-07-28_t00-59-52/"
dataset="anitamaxvim/jigsaw-toxic-comments"
root_dir="./"


if [[ $1 == *ignore* || $2 == *ignore* ]]; then
    echo Not calculating source
else
    # First we need to make the activations
    echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
        dataset=$dataset\
        max_samples=$max_samples\
        temperature=0\
        top_p=1\
        model_name="${root_dir}${model}"\
        $1 $2 $3 $4 $5 $6 $7 $8
    CUDA_VISIBLE_DEVICES=$cuda_devices python3 huggingface_collect_actvs.py\
        dataset=$dataset\
        temperature=0\
        top_p=1\
        max_samples=$max_samples\
        model_name="${root_dir}${model}"\
        $1 $2 $3 $4 $5 $6 $7 $8

fi


# Next we need to try running mas on the generated activations to ensure
# that a mapping can be learned

echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
    n_train_samples=500\
    n_valid_samples=100\
    n_units=100000\
    source_files="${root_dir}${model},${root_dir}${model}"\
    $1 $2 $3 $4 $5 $6 $7 $8
CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
    n_train_samples=500\
    n_valid_samples=100\
    n_units=100000\
    source_files="${root_dir}${model},${root_dir}${model}"\
    $1 $2 $3 $4 $5 $6 $7 $8
