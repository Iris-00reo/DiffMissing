#!/bin/bash
cd ../src

python_script="main.py"

scratch=True
cuda='cuda'
cuda_num="0"
dataset='PEMSBAY'
node_num=325
seq_len=12
missing_ratio=0.25
dataset_path="../datasets/$dataset/"

if [ $scratch = True ]; then
    log_path="../logs/scratch"
else
    log_path="../logs/test"
fi

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

for ((i=2024; i<=2028; i++))
do
    seed=$i

    echo "Running iteration with seed $seed on device cuda:$cuda_num"

    if [ $scratch = True ]; then
        nohup python -u $python_script \
            --scratch \
            --device $cuda \
            --cuda_num $cuda_num \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --node_num $node_num \
            --missing_ratio $missing_ratio \
            > $log_path/${dataset}/ms${missing_ratio}_seed${seed}.log 2>&1 &
    else
        nohup python -u $python_script \
            --device $cuda \
            --cuda_num $cuda_num \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --node_num $node_num \
            --missing_ratio $missing_ratio \
            --checkpoint_path $checkpoint_path \
            > $log_path/${dataset}/ms${missing_ratio}_seed${seed}.log 2>&1 &
    fi

    wait
    echo ""
done
