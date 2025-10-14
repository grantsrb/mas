#!/bin/bash/

model_names=("google/vit-base-patch16-224,google/vit-base-patch16-224")
lr="0.0001"
cuda_devices=7

for batch_size in 512
do
    echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 compare_vision_models.py\
        model_names=$model_names\
        subspace_size=800\
        mas_lr=$lr\
        mas_batch_size=$batch_size\
        $1 $2 $3 $4 $5 $6 $7 $8
    #CUDA_VISIBLE_DEVICES=$cuda_devices python3 compare_vision_models.py\
    #    model_names=$model_names\
    #    subspace_size=800\
    #    mas_lr=$lr\
    #    mas_batch_size=$batch_size\
    #    $1 $2 $3 $4 $5 $6 $7 $8
done

#for batch_size in 512
#do
#    echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 compare_vision_models.py\
#        model_names=$model_names\
#        subspace_size=800\
#        mas_lr=$lr\
#        mas_batch_size=$batch_size\
#        model_stitch=True\
#        $1 $2 $3 $4 $5 $6 $7 $8
#    CUDA_VISIBLE_DEVICES=$cuda_devices python3 compare_vision_models.py\
#        model_names=$model_names\
#        subspace_size=800\
#        mas_lr=$lr\
#        model_stitch=True\
#        mas_batch_size=$batch_size\
#        $1 $2 $3 $4 $5 $6 $7 $8
#done
