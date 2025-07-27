#!/bin/bash/

cuda_devices=1,2,3,5,7
mtx_types="RotationMatrix"
arg1="n_train_samples=1000"
arg2="n_valid_samples=500"
arg3="layers=model.layers.0"
arg4="incl_empty_varbs=False"
arg5="batch_size=32"
arg6="cl_directions=None"
root_dir="/data2/grantsrb/mas_finetunings/"

main_model="nontoxic_anitamaxvim-jigsaw-toxic-comments_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t13-16-04/srcactvs_anitamaxvim-jigsaw-toxic-comments_both_n400_d2025-07-27_t09-38-26.pt"

other_models=( "nontoxic_anitamaxvim-jigsaw-toxic-comments_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t13-16-04/srcactvs_anitamaxvim-jigsaw-toxic-comments_both_n400_d2025-07-27_t09-38-26.pt"

"nontoxic_Anthropic-hh-rlhf_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t16-31-08/srcactvs_anitamaxvim-jigsaw-toxic-comments_both_n400_d2025-07-27_t09-33-36.pt"

"nontoxic_lmsys-toxic-chat_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t21-17-25/srcactvs_anitamaxvim-jigsaw-toxic-comments_both_n400_d2025-07-27_t09-43-24.pt"

"toxic_anitamaxvim-jigsaw-toxic-comments_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t11-40-08/srcactvs_anitamaxvim-jigsaw-toxic-comments_both_n400_d2025-07-27_t09-41-03.pt"

"toxic_Anthropic-hh-rlhf_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t14-53-43/srcactvs_anitamaxvim-jigsaw-toxic-comments_both_n400_d2025-07-27_t09-35-54.pt"

"toxic_lmsys-toxic-chat_deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/run_d2025-07-26_t18-08-00/srcactvs_anitamaxvim-jigsaw-toxic-comments_both_n400_d2025-07-27_t09-45-47.pt"
)

for other_model in ${other_models[@]}
do
    echo CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
        $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $1 $2 $3 \
        source_files="${main_model},${other_model}"
    CUDA_VISIBLE_DEVICES=$cuda_devices python3 lang_mas.py\
        $arg1 $arg2 $arg3 $arg4 $arg5 $arg6 $1 $2 $3 \
        source_files="${root_dir}${main_model},${root_dir}${other_model}"
done
