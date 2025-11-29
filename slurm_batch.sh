#! /bin/bash

#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs/nf_6D_ag.out

./.venv/bin/python3 -u -m scripts.benchmark_6D_ag
