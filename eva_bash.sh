#user/bin/env bash

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=1214 evaluation.py \
                --load_8bit     \
                --base_model './hf_ckpt'     \
                --lora_weights './lora-alpaca-reranker-100k-bm25-gpt' \
                --prompt_template 'reranker' \
                --data_list 'nfc' > logs/log_gpu2_100_nfc_debug.log 2>&1 &