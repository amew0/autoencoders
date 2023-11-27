#!/bin/bash
#SBATCH --account=kunf0007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=diff-auto-vscode
#SBATCH --time=0:59
#SBATCH --partition=prod
#SBATCH --output=./o-%j.out

path="./o-"
j=$SLURM_JOB_ID
original_filename="${path}${j}.out"
new_filename="${path}${1}.out"
mv "$original_filename" "$new_filename"

# module purge
# module load miniconda/3

# conda activate eit
echo $1
echo $2
# python -u eit.py $1
