#!/bin/bash

learning_rates=(16.0 20.0) 
betas=(4.0 5.0)
for beta in "${betas[@]}"; do
for lr in "${learning_rates[@]}"; do
        python -u main.py --learning-rate $lr --beta $beta --output-size 3 --batch-dim 3 > results.txt
    done
done
