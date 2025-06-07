import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import os
from random import randint

# --- Setup environment
global_scratch = os.environ['GLOBALSCRATCH']
print(f"Global scratch: \n\t{global_scratch}")

run = wandb.init(
    project='LLaMA-3.2-3b-it Fine-tuning on BitextCS', 
    job_type="training", 
    anonymous="allow"
)

# --- Model configuration
model_id = "meta-llama/Llama-3.2-3B-Instruct"
new_model = f"{model_id.split('/')[1]}-bitext-cs-q8"
output_dir = f"{global_scratch}/{new_model}"
print(f"Output dir: \n\t{output_dir}")

bitext_ds = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# --- Precision based on GPU
torch_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
attn_implementation = "eager"  # Set to "flash_attention_2" if available

# --- Quantization config
model_kwargs = dict(
    torch_dtype=torch_dtype,
    device_map="auto",
    attn_implementation=attn_implementation,
    quantization_config=BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )
)

# --- Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Set pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

# --- Prompt template
system_prompt = "You are a top-rated customer service agent named John. Be polite to customers and answer all their questions."

def format_row(row):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]}
        ]
    }

# --- Load and prepare dataset
dataset = load_dataset(bitext_ds, split="train")
# dataset = dataset.shuffle().select(range(500))
dataset = dataset.shuffle()
dataset = dataset.map(format_row, remove_columns=dataset.features, batched=False, num_proc=28)
dataset = dataset.train_test_split(test_size=0.2)

print(dataset["train"][0])

# --- PEFT config for LLaMA 3
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
    modules_to_save=["lm_head", "embed_tokens"]
)

# --- Training config
args = SFTConfig(
    output_dir=output_dir,
    max_seq_length=512,
    packing=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=torch_dtype == torch.float16,
    bf16=torch_dtype == torch.bfloat16,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=True,
    hub_model_id=new_model,
    report_to="wandb",
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": True,
    }
)

# --- Trainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer
)

torch.cuda.empty_cache()
trainer.train()
print("Training finished")

# --- Save adapter model
trainer.save_model()
del model, trainer
torch.cuda.empty_cache()

# --- Merge adapter and base
print("Merging adapter and base model")
base_model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
peft_model = PeftModel.from_pretrained(base_model, output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{output_dir}_adapter_fused", safe_serialization=True, max_shard_size="2GB")

tokenizer.save_pretrained(f"{output_dir}_adapter_fused")

# --- Push to hub
print("Pushing model to hub")
merged_model.push_to_hub(new_model)
tokenizer.push_to_hub(new_model)
print("Pushed model to hub")

# --- Load and test
print("Loading merged model for testing")
model = AutoModelForCausalLM.from_pretrained(
    f"theflyingrahul/{new_model}",
    device_map="auto",
    torch_dtype=torch_dtype,
    attn_implementation=attn_implementation,
)
tokenizer = AutoTokenizer.from_pretrained(f"theflyingrahul/{new_model}", trust_remote_code=True)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

rand_idx = randint(0, len(dataset["test"]))
test_sample = dataset["test"][rand_idx]
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

prompt = tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)
print(f"Prompt:\n{prompt}")

outputs = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=False,
    temperature=0.1,
    top_k=50,
    top_p=0.1,
    eos_token_id=stop_token_ids,
    disable_compile=True
)

print(f"Original Answer:\n{test_sample['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text']}")
