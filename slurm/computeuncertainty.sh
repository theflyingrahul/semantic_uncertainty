#!/bin/bash
#
#SBATCH --output=%j_%x.out
#SBATCH --partition=batch          # Lyra (or --partition=gpu for Hercules / Dragon2)

# --- resources ------------------------------------------------------
#SBATCH --gpus=1                   # one GPU per node on Lyra
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=0-4:00:00             # walltime D‑HH:MM:SS

# --- environment ----------------------------------------------------
module purge
module load Python
module load CUDA
source ../venv/bin/activate

# --- run ------------------------------------------------------------
python compute_uncertainty_measures.py --metric=custom_llm --custom_metric_model_name=Llama-3.2-3B-Instruct --entailment_model=Llama-3.2-3B-Instruct --no-compute_accuracy_at_all_temps --eval_wandb_runid="${WANDB_ID}" --entity=theflyingrahul-indian-institute-of-management-bangalore --restore_entity_eval=theflyingrahul-indian-institute-of-management-bangalore