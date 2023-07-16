#user/bin/env bash

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=1214 evaluation.py \
                --load_8bit     \
                --base_model './llama-7b-hf'     \
                --lora_weights './models/wtf/rank_weight' \
                --prompt_template 'alpaca' \
                --data_list 'dl21'
#                > logs/log_gpu1_nfc_debug.log 2>&1 &