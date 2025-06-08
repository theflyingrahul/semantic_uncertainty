#!/bin/bash

# Define a bash associative array (requires bash 4+)
declare -A RUNS=(
    ["llama_t0.1"]="xo87ib73"
    ["llama_t0.3"]="2pi0w7eb"
    ["llama_t0.5"]="apd4tp5g"
    ["llama_t0.7"]="2ru2bvrn"
    ["llama_t1"]="s1rjsb8n"
    ["llamaft_t0.1"]="2qxnxd3d"
    ["llamaft_t0.3"]="g2wdwzh9"
    ["llamaft_t0.5"]="b4x6333i"
    ["llamaft_t0.7"]="aqv04kes"
    ["llamaft_t1"]="0rrsqndw"
    ["gemma_t0.1"]="6tgrl1li"
    ["gemma_t0.3"]="o8y0gpm4"
    ["gemma_t0.5"]="gv27eytu"
    ["gemma_t0.7"]="x2aztdd8"
    ["gemma_t1"]="e53f59kj"
    ["gemmaft_t0.1"]="ariwy32a"
    ["gemmaft_t0.3"]="dkdbklel"
    ["gemmaft_t0.5"]="m37a32w8"
    ["gemmaft_t0.7"]="pw9h5cau"
    ["gemmaft_t1"]="dgi7kj13"
)

for key in "${!RUNS[@]}"; do
    WANDB_ID="${RUNS[$key]}"

    # Extract MODEL_ID and TEMPERATURE
    MODEL_ID="${key%%_t*}"                # Part before _t
    TEMPERATURE="${key#*_t}"             # Part after _t

    echo "Submitting: MODEL_ID=$MODEL_ID, TEMPERATURE=$TEMPERATURE, WANDB_ID=$WANDB_ID"

    sbatch --job-name=${MODEL_ID}_fastentail_t${TEMPERATURE}_nocontext \
           --export=WANDB_ID=${WANDB_ID} \
           computeuncertainty.sh
done
