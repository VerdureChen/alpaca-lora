#user/bin/env bash
WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 generate.py \
                --load_8bit     \
                --base_model './hf_ckpt'     \
                --lora_weights './lora-alpaca-reranker-100k-bm25-gpt' \
                --prompt_template 'reranker'