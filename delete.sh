#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=import
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=kunf0007
#SBATCH --output=i.%j.out
#SBATCH --error=i.%j.err
 
module purge
module load miniconda/3
module load cuda/11.4
module load gcc/9.3

nvidia-smi
