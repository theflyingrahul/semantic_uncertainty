#!/bin/bash

# Define a bash associative array (requires bash 4+) (The below keys are the 512 token runs. Llama t1 is missing.)
declare -A RUNS=(
    ["mistral_t0.1"]="3yrkf7u1"
    ["mistral_t0.3"]="b8c7h68g"
    ["mistral_t0.5"]="vb4pa3si"
    ["mistral_t0.7"]="lvu5ri02"
    ["mistral_t1"]="zm5046x2"

    ["mistralft_t0.1"]="794l9t4t"
    ["mistralft_t0.3"]="2tky4jqm"
    ["mistralft_t0.5"]="49xqdys6"
    ["mistralft_t0.7"]="nlrikrv6"
    ["mistralft_t1"]="fd9uv5dz"

    ["llama_t0.1"]="kqzuuyve"
    ["llama_t0.3"]="1es6hap0"
    ["llama_t0.5"]="m2url9ia"
    ["llama_t0.7"]="r4krcf0o"

    ["llamaft_t0.1"]="8ejuek8e"
    ["llamaft_t0.3"]="yphad4mz"
    ["llamaft_t0.5"]="kw0ou5b4"
    ["llamaft_t0.7"]="m5c9y0ba"
    ["llamaft_t1"]="4td4sor8"

    ["gemma_t0.1"]="9r9qpofk"
    ["gemma_t0.3"]="70pglysi"
    ["gemma_t0.5"]="budi27cq"
    ["gemma_t0.7"]="k8cn1dij"
    ["gemma_t1"]="mtk9ebyy"

    ["gemmaft_t0.1"]="jgaau079"
    ["gemmaft_t0.3"]="p0gnnxjb"
    ["gemmaft_t0.5"]="13005qc6"
    ["gemmaft_t0.7"]="whv77roe"
    ["gemmaft_t1"]="t966qn7b"
)


for key in "${!RUNS[@]}"; do
    WANDB_ID="${RUNS[$key]}"

    # Extract MODEL_ID and TEMPERATURE
    MODEL_ID="${key%%_t*}"                # Part before _t
    TEMPERATURE="${key#*_t}"             # Part after _t

    echo "Submitting: MODEL_ID=$MODEL_ID, TEMPERATURE=$TEMPERATURE, WANDB_ID=$WANDB_ID"

    sbatch --job-name=${MODEL_ID}_fastentail_t${TEMPERATURE} \
           --export=MODEL_ID=${MODEL_ID},TEMPERATURE=${TEMPERATURE},WANDB_ID=${WANDB_ID} \
           fast_entailment.sh
done
