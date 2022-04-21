#!/bin/bash

sigma=${1:-0.5}
dataset_path=${2:-../data/TVSum/eccv16_dataset_tvsum_google_pool5.h5}
splits_path=${3:-../data/splits/tvsum_splits.json}
echo sigma: $sigma
echo dataset_path: $dataset_path
echo splits_path : $splits_path

for i in {0..4}; do
    python main.py --split_index $i --regularization_factor $sigma --dataset_path $dataset_path --splits_path $splits_path
done
