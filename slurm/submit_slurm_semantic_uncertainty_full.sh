#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: Temperature argument is required."
  echo "Usage: $0 <temperature>"
  exit 1
fi

TEMP=$1

MODEL_IDS=(
  "theflyingrahul/Mistral-7B-Instruct-v0.2-bitext-cs-q8"
  "Mistral-7B-Instruct-v0.2"
  "theflyingrahul/Llama-3.2-3B-Instruct-bitext-cs-q8"
  "Llama-3.2-3B-Instruct"
  "google/gemma-3-4b-it"
  "theflyingrahul/gemma-3-4b-it-bitext-cs-q8"
)

MODEL_TINYS=(
  "mistralft"
  "mistral"
  "llamaft"
  "llama"
  "gemma"
  "gemmaft"
)

for i in "${!MODEL_IDS[@]}"; do
  MODEL_ID="${MODEL_IDS[$i]}"
  MODEL_TINY="${MODEL_TINYS[$i]}"
  
  echo "Submitting: MODEL_ID=$MODEL_ID, TEMPERATURE=$TEMP"
  
  sbatch --job-name=${MODEL_TINY}_fullSE_t${TEMP} \
         --export=MODEL_ID=${MODEL_ID},TEMPERATURE=${TEMP} \
         slurm_semantic_uncertainty_full.sh
done
