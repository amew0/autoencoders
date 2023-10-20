#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=autoencoders
#SBATCH --time=71:59:59
#SBATCH --partition=prod
#SBATCH --account=kunf0007
#SBATCH --output=./0016/%A_%a.out
#SBATCH --error=./0016/%A_%a.err
#SBATCH --array=1-8

module purge
module load miniconda/3

echo $SLURM_ARRAY_TASK_ID
yml=$(head -n $SLURM_ARRAY_TASK_ID $1 | tail -n 1)

python -u eit_0016.py $yml

