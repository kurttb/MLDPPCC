#!/bin/bash
#SBATCH --job-name=final_a1_tune
#SBATCH --account=ners590s002w26_class
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1500m
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mail-user=kurttb@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=final_a1_tune_%j.log
#SBATCH --error=final_a1_tune_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate tf-blackwell

cd "/home/burt/Documents/Local Coursework Folder/NERS 590 Local Work/Project/MLDPPCC"

NOTEBOOK="Final_A1_Tune_Singles_MultiTaskerEncoder_HyperParameter.ipynb"
RUN_LOG="Final_A1_Tune_Singles_MultiTaskerEncoder_HyperParameter_run.log"

echo "Starting notebook execution: ${NOTEBOOK}"
echo "Notebook log: ${RUN_LOG}"

nohup jupyter nbconvert --to notebook --execute --inplace "${NOTEBOOK}" > "${RUN_LOG}" 2>&1