#! /bin/bash

#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm_benchmark_&j.out

./.venv/bin/python3 -u -m scripts.benchmark_6D_ag