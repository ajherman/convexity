#!/bin/bash -l

#SBATCH --job-name=main
#SBATCH --time 10:00:00
#SBATCH -N 15
#SBATCH -p shared-gpu
#module load miniconda3
#source activate /vast/home/ajherman/miniconda3/envs/pytorch

hidden1s=(256 384)
hdden2s=(256 384)
learning_rates=(0.2 0.5 1.0 2.0) 
betas=(0.2 0.5 1.0 2.0 5.0)
# mr=(0.5 1.0)
lam=(2.0 1.0)
for beta in "${betas[@]}"; do
for lr in "${learning_rates[@]}"; do
# for mr in "${mr[@]}"; do
for l in "${lam[@]}"; do
for hidden1 in "${hidden1s[@]}"; do
for hidden2 in "${hidden2s[@]}"; do
        # name="lr_${lr}_beta_${beta}"
        name="lr_${lr}_beta_${beta}_mr_${mr}_lam_${l}_h1_${hidden1}_h2_${hidden2}"
        srun -N 1 -n 1 -c 6 -o $name.out --open-mode=append ./main_wrapper.sh --output-dir $name --learning-rate $lr --beta $beta --mr 0.5 --lam $l --hidden1-size $hidden1 --hidden2-size $hidden2 --init zeros &
        # python3 -u main.py --learning-rate $lr --beta $beta --input-size 784 --output-size 10 --batch-dim 50 --init random --n-iters 3000 >> results.txt 
done
# done
done
done
done
done

# # # From paper
# python3 -u main.py --learning-rate 5.0 --beta 2.0 --input-size 784 --hidden1-size 200 --hidden2-size 200 --output-size 10 --mr 0.5 --lam 2.0 --batch-dim 100 --init zeros --dataset mnist --n-iters 2000 >> results.txt
# python3 -u main.py --learning-rate 1.0 --beta 0.2 --input-size 784 --hidden1-size 256 --hidden2-size 256 --output-size 10 --mr 0.5 --lam 2.0 --batch-dim 200 --init zeros --dataset mnist --n-iters 2000 --make-tsne >> results.txt
# python3 -u main.py --learning-rate 1.0 --beta 1.0 --input-size 784 --hidden1-size 256 --hidden2-size 256 --output-size 10 --mr 0.5 --lam 2.0 --batch-dim 50 --init zeros --dataset mnist --n-iters 10000 --make-tsne >> results.txt
# srun -N 1 -n 1 -c 6 -o results.out --open-mode=append ./main_wrapper.sh  --learning-rate $lr --beta $beta --mr $mr --lam $l --input-size 784 --output-size 10 --hidden1-size 500 --hidden2-size 500 --batch-dim 200 --init zeros --n-iters 2000 &

# python3 -u main.py --learning-rate 0.1 --n-epoch 5 --hidden1-size 400 --hidden2-size 400 --output-file errors --make-tsne >> results.txt
