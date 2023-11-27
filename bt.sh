#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=diff-auto-vscode
#SBATCH --time=59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=kunf0007
#SBATCH --output=./output/img/bt-%j.out

path="./output/img/bt-"
j=$SLURM_JOB_ID
original_filename="${path}${j}.out"
new_filename="${path}${1}.out"
mv "$original_filename" "$new_filename"

module purge
module load miniconda/3

conda activate eit
echo $j
echo $1
echo $2

python -u bt.py
