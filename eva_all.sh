#!/bin/bash

datasets=("dl19" "dl20" "dbpedia" "nfc" "covid"  "robust04" "scifact" "news" "signal" "touche")
gpu_num=(0 1 2 3 4 5 6 7)
num_gpus=8
num_datasets=${#datasets[@]}

datasets_per_gpu=1
#extra_datasets=$(( num_datasets - num_gpus ))

for ((gpu=0; gpu<num_gpus; gpu++)); do
    start_index=$(( gpu * datasets_per_gpu ))
    data_list="${datasets[$start_index]}"

    # The 7th and 8th GPU run an extra dataset
    if (( gpu >= num_gpus - num_gpus )); then
        extra_index=$(( start_index + num_gpus ))
        data_list="${data_list},${datasets[$extra_index]}"
    fi

    master_port=$(( 1224 + gpu ))
    log_file="logs/full_log/log_gpu${gpu}_${data_list}_fulldbg.log"

    # shellcheck disable=SC2089
    gpu_number=${gpu_num[$gpu]}
    command="CUDA_VISIBLE_DEVICES=$gpu_number torchrun --nproc_per_node=1 --master_port=$master_port evaluation.py \
        --load_8bit \
        --base_model './models/reranker-full-alpaca' \
        --lora_weights './lora-alpaca-reranker-100k-bm25-gpt' \
        --prompt_template 'reranker' \
        --data_list '$data_list' > $log_file 2>&1 &"
    echo $command
    eval $command
done
