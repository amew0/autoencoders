#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=v2lr-auto-vscode
#SBATCH --time=4:59:59
#SBATCH --partition=prod
#SBATCH --account=kunf0007
#SBATCH --output=./output/v2lr/v2lr-%j.out

path="./output/v2lr/v2lr-"
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
python -u v2lr.py $1