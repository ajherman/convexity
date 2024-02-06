#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 8
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

learning_rates=(0.1 0.2 0.5 1.0 2.0 5.0 10.0) 
betas=(0.1 0.2 0.5 1.0 2.0 5.0 10.0)
mr=(0.05 0.1 0.2 0.5)
lam=(0.5 1.0 2.0 5.0 10.0 20.0)
for beta in "${betas[@]}"; do
for lr in "${learning_rates[@]}"; do
for mr in "${mr[@]}"; do
for l in "${lam[@]}"; do
        name="lr_${lr}_beta_${beta}"
        srun -N 1 -n 1 -c 6 -o $name.out --open-mode=append ./main_wrapper.sh  --learning-rate $lr --beta $beta --mr $mr --lam $l --input-size 784 --output-size 10 --hidden1-size 200 --hidden2-size 200 --batch-dim 100 --init zeros --n-iters 2000 &
        # python3 -u main.py --learning-rate $lr --beta $beta --input-size 784 --output-size 10 --batch-dim 50 --init random --n-iters 3000 >> results.txt 
done
done
done
done

# # # From paper
# # python3 -u main.py --learning-rate 10.0 --beta 10.0 --input-size 784 --hidden1-size 256 --hidden2-size 256 --output-size 10 --mr 0.1 --batch-dim 20 --init zeros --dataset mnist --n-iters 2000 --free-steps 25 --nudge-steps 5 >> results.txt
# python3 -u main.py --input-size 784 --output-size 10 --init random --dataset mnist  >> results.txt