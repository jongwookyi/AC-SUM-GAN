#!/bin/bash

# example usage: bash evaluate_exp.sh <path_to_experiment> TVSum sigma0.5 avg
base_path=${1:-../exp1}
dataset_name=${2:-TVSum}
exp_name=${3:-sigma0.5}
eval_method=${4:-avg}

# change this path if you use different structure for your directories inside the experiment
exp_path="$base_path/$dataset_name/$exp_name"

for i in {0..4}; do 
    logs_path="$exp_path/logs/split$i"
    python exportTensorFlowLog.py $logs_path $logs_path
    results_path="$exp_path/results/split$i"
    python compute_fscores.py $results_path $dataset_name $eval_method
done
