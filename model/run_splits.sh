#!/bin/bash

sigma=${1:-0.5}
echo sigma $sigma

for i in {0..4}; do
    python main.py --split_index $i --regularization_factor $sigma
done
