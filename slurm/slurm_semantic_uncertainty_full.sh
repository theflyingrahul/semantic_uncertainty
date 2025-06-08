#!/bin/bash
#
#SBATCH --output=%j_%x.out
#SBATCH --partition=batch          # Lyra (or --partition=gpu for Hercules / Dragon2)

# --- resources ------------------------------------------------------
#SBATCH --gpus=1                   # one GPU per node on Lyra
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=0-23:59:00             # walltime D‑HH:MM:SS

# --- environment ----------------------------------------------------
module purge
module load Python
module load CUDA
source ../venv/bin/activate

# --- run ------------------------------------------------------------
echo "Running with MODEL_ID=${MODEL_ID}, TEMPERATURE=${TEMPERATURE}"

python generate_answers.py --model_name="${MODEL_ID}" --dataset=bitext_cs --num_few_shot=0 --model_max_new_tokens=512 --brief_prompt=chat --metric=custom_llm --custom_metric_model_name=Llama-3.2-3B-Instruct --entailment_model=Llama-3.2-3B-Instruct --no-compute_accuracy_at_all_temps --temperature=${TEMPERATURE} --alt_entail_prompt