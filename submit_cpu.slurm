#!/bin/bash

#SBATCH --job-name=hyperparameter_classificator
#SBATCH --partition=shared

## 3 day max run time for public partitions, except 4 hour max runtime for the sandbox partition
#SBATCH --time=3-00:00:00 ## time format is DD-HH:MM:SS

#SBATCH --cpus-per-task=24
#SBATCH --mem=30000## max amount of memory per node you require

#SBATCH --error=cpu_hptrain-%A.err ## %A - filled with jobid
#SBATCH --output=cpu_hptrain-%A.out ## %A - filled with jobid

## Useful for remote notification
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT_80
#SBATCH --mail-user=victorgc@hawaii.edu

## All options and environment variables found on schedMD site: http://slurm.schedmd.com/sbatch.html
ml purge
ml lang/Anaconda3
source activate tf
python hp_train.py

