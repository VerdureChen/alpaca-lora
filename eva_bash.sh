#user/bin/env bash

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=1294 evaluation.py \
                --load_8bit     \
                --base_model './models/reranker-full-alpaca'     \
                --lora_weights './lora-alpaca-reranker-100k-bm25-gpt' \
                --prompt_template 'reranker' \
                --data_list 'news'