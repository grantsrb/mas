
datasets=("svhn" "mnist" "fashion-mnist") #"cifar10" 
#model_names=("microsoft/resnet-18,microsoft/resnet-18" "google/vit-base-patch16-224,google/vit-base-patch16-224")
model_names=("google/vit-base-patch16-224,google/vit-base-patch16-224")
mas_lr=0.0001
batch_size=512
cuda_devices=7

for model_name in ${model_names[@]}
do
    for dataset in ${datasets[@]}
    do
        for model_stitch in "True" "False"
        do
        echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 compare_vision_models.py\
            dataset_name=$dataset\
            model_names=$model_name\
            model_stitch=$model_stitch\
            mas_batch_size=$batch_size\
            mas_lr=$mas_lr\
            $1 $2 $3 $4 $5 $6 $7 $8
        CUDA_VISIBLE_DEVICES=$cuda_devices python3 compare_vision_models.py\
            dataset_name=$dataset\
            model_names=$model_name\
            model_stitch=$model_stitch\
            mas_batch_size=$batch_size\
            mas_lr=$mas_lr\
            $1 $2 $3 $4 $5 $6 $7 $8
        done
    done
done
