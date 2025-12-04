#! /bin/bash

#SBATCH --time=23:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sos_12D_ag.out

./.venv/bin/python3 -u -m scripts.benchmark_12D_ag
