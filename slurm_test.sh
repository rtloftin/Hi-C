#!/bin/sh

#SBATCH --partition=influence
#SBATCH --qos=short
#SBATCH --time=00:01:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu=512M
#SBATCH --job-name=slurm_test
#SBATCH --output=./slurm_output/%j.out

echo "launching 4 tasks"
srun -n1 --output=./slurm_output/%J.out echo "test 1" &
srun -n1 --output=./slurm_output/%J.out echo "test 2" &
srun -n1 --output=./slurm_output/%J.out echo "test 3" &
srun -n1 --output=./slurm_output/%J.out echo "test 4" &
wait