#!/bin/bash
#SBATCH --job-name=ecg_full_study
#SBATCH --output=logs/full_study_%j.out
#SBATCH --error=logs/full_study_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

cd "$(dirname "$0")/.."
source venv/bin/activate
mkdir -p logs
python main.py