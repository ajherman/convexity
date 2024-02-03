#!/bin/bash

learning_rates=(100.0 200.0) 
betas=(2.0 3.0)
for beta in "${betas[@]}"; do
for lr in "${learning_rates[@]}"; do
        python -u main.py --learning-rate $lr --beta $beta --output-size 20 --batch-dim 20 > results.txt
    done
done
