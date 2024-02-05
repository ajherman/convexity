#!/bin/bash

learning_rates=(0.5 1.0 2.0 4.0 8.0 16.0) 
betas=(0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8)
for beta in "${betas[@]}"; do
for lr in "${learning_rates[@]}"; do
        name="lr_${lr}_beta_${beta}"
#  <<<<<<< HEAD
        srun -N 1 -n 1 -c 6 -o $name.out --open-mode=append ./main_wrapper.sh  --learning-rate $lr --beta $beta --output-size 20 --batch-dim 20 --init zeros --n-iters 20000 &
# =======
#       srun -N 1 -n 1 -c $cores -o $name.out --open-mode=append ./main_wrapper.sh  --learning-rate $lr --beta $beta --output-size 20 --batch-dim 20 --init zeros --n-iters 20000 &
# >>>>>>> 90feec8c35122d3b50f9bdd4bc450d9cb4119f2a
        # python -u main.py --learning-rate $lr --beta $beta --output-size 20 --batch-dim 20 --init zeros --n-iters 40000 > results.txt
    done
done
