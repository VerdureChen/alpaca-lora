#!/bin/bash

datasets=("dl19" "dl20" "covid" "touche" "news" "signal" "dbpedia" "scifact" "nfc" "robust04")
num_gpus=8

for ((gpu=0; gpu<num_gpus; gpu++)); do
    if (( gpu < 2 )); then
        start_index=$(( gpu * 2 ))
        end_index=$(( start_index + 1 ))
    else
        start_index=$(( 4 + (gpu - 2) ))
        end_index=$start_index
    fi

    data_list=""
    for ((i=start_index; i<=end_index; i++)); do
        if [ -z "$data_list" ]; then
            data_list="${datasets[$i]}"
        else
            data_list="${data_list},${datasets[$i]}"
        fi
    done

    master_port=$(( 1224 + gpu ))
    log_file="logs/log_gpu${gpu}_100_${data_list}.log"

    # shellcheck disable=SC2089
    command="CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$master_port evaluation.py \
        --load_8bit \
        --base_model './hf_ckpt' \
        --lora_weights './lora-alpaca-reranker-100k-bm25-gpt' \
        --prompt_template 'reranker' \
        --data_list '$data_list' > $log_file 2>&1 &"
    echo $command
    eval $command
done