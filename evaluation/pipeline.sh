#!/bin/bash

path_to_experiment=${1:-../exp1}
dataset_name=${2:-TVSum}
eval_method=${3:-avg}

# Run the evaluation script with the right arguments to compute the F-Scores and extract the loss values for each value of the regularization factor (sigma)
for sigma in $(seq 0.1 0.1 1); do
    exp_name=sigma$sigma
    bash evaluate_exp.sh $path_to_experiment $dataset_name $exp_name $eval_method
done

# Run the script that chooses the best epoch and defines the final F-Score value
python choose_best_epoch.py $path_to_experiment $dataset_name
