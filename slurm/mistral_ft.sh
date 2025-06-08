#!/bin/bash
#
#SBATCH --job-name=mistral_ft_bitext_4b_q8_full
#SBATCH --output=%j_%x.out
#SBATCH --partition=batch          # Lyra (or --partition=gpu for Hercules / Dragon2)

# --- resources ------------------------------------------------------
#SBATCH --gpus=1                   # one GPU per node on Lyra
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=0-6:00:00             # walltime D‑HH:MM:SS

# --- environment ----------------------------------------------------
module purge
module load Python
module load CUDA
source venv/bin/activate

# --- run ------------------------------------------------------------
python mistral_ft.py