#user/bin/env bash
#python -m torch.distributed.launch  \
#--nproc_per_node=8 \
WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
            --base_model='./llama-7b-hf' \
            --data_path './alpaca_data_gpt4.json' \
            --output_dir './lora-alpaca' \
            --num_epochs=3 \
            --cutoff_len=512 \
            --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
            --lora_r=16 \
            --micro_batch_size=4 \
            --batch_size=256 \
            --wandb_project 'lora-alpaca' 