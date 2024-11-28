#!/usr/bin/bash

#SBATCH -J BBDM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 4-0
#SBATCH -o logs/slurm-%A.out

pwd
which python
thom08
python main.py --config /data/thom08/rental_seraph_lwg/BBDM/configs/Template-LBBDM-f4.yaml --train --sample_at_start --save_top --gpu_ids 0 
exit 0