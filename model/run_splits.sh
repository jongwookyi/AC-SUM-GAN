#!/bin/bash

sigma=${1:-0.5}
dataset=${2:-TVSum}
dataset_path=${3:-../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5}
splits_path=${4:-../data/splits/tvsum_splits.json}

echo dataset: $dataset
echo dataset path: $dataset_path
echo splits path : $splits_path
echo sigma: $sigma

for i in {0..4}; do
    python main.py --split_index $i --regularization_factor $sigma --dataset $dataset --dataset_path $dataset_path --splits_path $splits_path
done
