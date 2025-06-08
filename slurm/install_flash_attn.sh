#!/bin/bash
#
#SBATCH --job-name=install_flash_attn
#SBATCH --output=%j_%x.out
#SBATCH --partition=batch          # Lyra (or --partition=gpu for Hercules / Dragon2)

# --- resources ------------------------------------------------------
#SBATCH --gpus=1                   # one GPU per node on Lyra
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=0-6:00:00             # walltime D‑HH:MM:SS

# --- environment ----------------------------------------------------
module purge
module load Python
module load CUDA
source venv/bin/activate

# --- run ------------------------------------------------------------
MAKEFLAGS="-j16" pip install flash_attn --no-build-isolation --use-pep517 --no-cache-dir