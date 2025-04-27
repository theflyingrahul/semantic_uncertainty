# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains the code necessary to reproduce the short-phrase and sentence-length experiments of the Nature submission *"Detecting Hallucinations in Large Language Models Using Semantic Entropy."*

## About This Fork

This is a fork of the original repository with modifications to enable execution on consumer-grade GPUs and additional features. Extensive documentation of the changes and adaptations is planned.

## Added Arguments

### `--second_gpu`
- Enables sharding across secondary low-power GPUs asymmetrically.
- **Note:** VRAM limits are currently hardcoded (0, 6GB, 6GB). A new argument to configure these limits is planned.

### `--metric=custom_llm`
- Allows the use of a custom LLM for computing accuracy (e.g., Llama 3.2).

### `--custom_metric_model_name`
- Complements the `--metric=custom_llm` argument. Specify the model name here.

## Running Independent Computations

To compute uncertainty measures independently, use the following command:

```bash
python compute_uncertainty_measures.py \
    --metric=custom_llm \
    --custom_metric_model_name=Llama-3.2-3B-Instruct \
    --entailment_model=Llama-3.2-3B-Instruct \
    --no-compute_accuracy_at_all_temps \
    --eval_wandb_runid=sddy15no \
    --entity=theflyingrahul-indian-institute-of-management-bangalore \
    --restore_entity_eval=theflyingrahul-indian-institute-of-management-bangalore
```

## Fixes and Additions from the Original Repository

1. **Whitespace Handling:**
     - Resolved an issue where additional whitespaces in LLM responses caused a `ValueError`. This was hotpatched by approximating the output window shift.

2. **Model and Dataset Support:**
     - Added support for additional models and datasets. *(TODO: Document this in detail.)*

3. **Demonstrations:**
     - Included demonstrations for:
         - Dissecting the generated `.pkl` files.
         - Analyzing stochasticity in outputs.

## TODOs
- Document the added model and dataset support.
- Add an argument to configure VRAM limits for `--second_gpu`.
