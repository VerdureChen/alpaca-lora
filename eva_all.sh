#!/bin/bash

datasets=("dl19" "dbpedia" "dl20" "nfc" "covid"  "robust04" "signal" "touche" "news" "scifact")
num_gpus=5
num_datasets=${#datasets[@]}

datasets_per_gpu=$(( (num_datasets + num_gpus - 1) / num_gpus ))

for ((gpu=0; gpu<num_gpus; gpu++)); do
    start_index=$(( gpu * datasets_per_gpu ))
    end_index=$(( start_index + datasets_per_gpu - 1 ))
    if (( end_index >= num_datasets )); then
        end_index=$(( num_datasets - 1 ))
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
    log_file="logs/log_gpu${gpu}_${data_list}.log"

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