#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=diff-auto-vscode
#SBATCH --time=71:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=kunf0007
#SBATCH --output=./output/diffimgs%j.out
#SBATCH --error=./output/diffimgs%j.err

module purge
module load miniconda/3

conda activate eit
echo $1
python -u diffimg.py $1
