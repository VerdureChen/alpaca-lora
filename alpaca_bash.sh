#user/bin/env bash
#python -m torch.distributed.launch  \
#--nproc_per_node=8 \
WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
            --base_model='./hf_ckpt' \
            --data_path './process_data/reranker_100k_bm25_2_gpt_ot.json' \
            --output_dir './models/lora-alpaca-reranker-100k-bm25-gpt-lr6' \
            --num_epochs=3 \
            --cutoff_len=2047 \
            --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
            --lora_r=16 \
            --micro_batch_size=1 \
            --batch_size=256 \
            --wandb_project 'lora-alpaca' \
            --prompt_template_name 'reranker' \
            --add_eos_token