#user/bin/env bash
WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 generate.py \
                --load_8bit     \
                --base_model './llama-7b-hf'     \
                --lora_weights './lora-alpaca' \
                --prompt_template 'reranker'