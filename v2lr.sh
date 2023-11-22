#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=v2lr-auto-vscode
#SBATCH --time=0:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=kunf0007
#SBATCH --output=./output/v2lr/v2lr-%j.out

module purge
module load miniconda/3

conda activate eit
echo $1
echo $2
python -u v2lr.py $1