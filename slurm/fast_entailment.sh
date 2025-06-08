#!/bin/bash
#
#SBATCH --output=%j_%x.out
#SBATCH --partition=batch          # Lyra (or --partition=gpu for Hercules / Dragon2)

# --- resources ------------------------------------------------------
#SBATCH --gpus=1                   # one GPU per node on Lyra
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=0-4:00:00             # walltime D‑HH:MM:SS

# --- environment ----------------------------------------------------
module purge
module load Python
module load CUDA
source venv/bin/activate

# --- run ------------------------------------------------------------
echo "Running with MODEL_ID=${MODEL_ID}, TEMPERATURE=${TEMPERATURE}, WANDB_ID=${WANDB_ID}"
# python test.py "${MODEL_ID}" "${TEMPERATURE}" "${WANDB_ID}"
python fast_entailment.py "${MODEL_ID}" "${TEMPERATURE}" "${WANDB_ID}"
