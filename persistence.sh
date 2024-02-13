#!/bin/bash

learning_rates=(1.0 0.1 0.01 0.001)
betas=(0.1 0.5 1.0 2.0 5.0)

for lr in "${learning_rates[@]}"; do
  for beta in "${betas[@]}"; do
    python3 -u persistence.py --learning-rate $lr --beta $beta 
  done
done
