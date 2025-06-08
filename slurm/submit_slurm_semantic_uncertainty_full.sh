#!/bin/bash

# MODEL_ID="theflyingrahul/Mistral-7B-Instruct-v0.2-bitext-cs-q8"
# MODEL_ID="Mistral-7B-Instruct-v0.2"
MODEL_ID="theflyingrahul/Llama-3.2-3B-Instruct-bitext-cs-q8"
# MODEL_ID="Llama-3.2-3B-Instruct"
# MODEL_ID="google/gemma-3-4b-it"
# MODEL_ID="theflyingrahul/gemma-3-4b-it-bitext-cs-q8"

# MODEL_TINY="mistralft"
# MODEL_TINY="mistral"
MODEL_TINY="llamaft"
# MODEL_TINY="llama"
# MODEL_TINY="gemma"
# MODEL_TINY="gemmaft"

TEMPERATURES=(0.1 0.3 0.5 0.7 1)

for TEMP in "${TEMPERATURES[@]}"; do
  echo "Submitting: MODEL_ID=$MODEL_ID, TEMPERATURE=$TEMP"
  
  sbatch --job-name=${MODEL_TINY}_fullSE_t${TEMP} \
         --export=MODEL_ID=${MODEL_ID},TEMPERATURE=${TEMP} \
         slurm_semantic_uncertainty_full.sh
done