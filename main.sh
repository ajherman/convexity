#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 15
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

learning_rates=(0.5 1.0 2.0 4.0 8.0 16.0) 
betas=(0.1 0.2 0.4 0.8 1.6 3.2 6.4 12.8)
for beta in "${betas[@]}"; do
for lr in "${learning_rates[@]}"; do
        name="lr_${lr}_beta_${beta}"
        srun -N 1 -n 1 -c 6 -o $name.out --open-mode=append ./main_wrapper.sh  --learning-rate $lr --beta $beta --output-size 20 --batch-dim 20 --init random --n-iters 20000 &
        # python -u --learning-rate $lr --beta $beta --output-size 20 --batch-dim 20 --init zeros --n-iters 20000 > results.txt &
    done
done
