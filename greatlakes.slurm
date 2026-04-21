#!/bin/bash
#SBATCH --job-name=c1_last32
#SBATCH --account=ners590s002w26_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1500m
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --mail-user=kurttb@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=c1_last32_%j.log
#SBATCH --error=c1_last32_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate tf-blackwell

cd "/home/burt/Documents/Local Coursework Folder/NERS 590 Local Work/Project/MLDPPCC"

python greatlakes.py