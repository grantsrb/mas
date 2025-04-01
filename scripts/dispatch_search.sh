#!/bin/bash
# Use this script to run the DAS experiments

exp_name="mas_dispatch"
gpus=( 0 1 2 3 4 5 6 7 8 9 )
model_path1="/mnt/fs2/grantsrb/mas_saves/big-multiobj/gru_size_devo_14_seed10_d_model48_n_layers1"
model_path2="/mnt/fs2/grantsrb/mas_saves/big-multiobj/gru_size_devo_19_seed11_d_model48_n_layers1"
config="configs/numequiv.yaml"
const_targ_inpt_id="const_targ_inpt_id=False"

echo Dispatching
cuda_idx=0

for fsr in "fsr=False" "fsr=True"
do
for n_units in "n_units=16" "n_units=24" "n_units=32" "n_units=40" "n_units=48"
do
   cuda=${gpus[$cuda_idx]}

   bash scripts/single_model_run.sh $cuda $exp_name $model_path1 $model_path2 $config $n_units $fsr $const_targ_inpt_id &

   cuda_idx=$((1+$cuda_idx))
   if [[ ${cuda_idx} == ${#gpus[@]} ]]; then
       cuda_idx=0
   fi
done
done





