import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig, pipeline
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from random import randint
import re
from trl import SFTConfig, SFTTrainer
import os

global_scratch = os.environ['GLOBALSCRATCH']
print(f"Global scratch: \n\t{global_scratch}")

run = wandb.init(
    project='Gemma-3-4b-it Fine-tuning on BitextCS', 
    job_type="training", 
    anonymous="allow"
)

attn_implementation = "eager" # Use "flash_attention_2" when running on Ampere or newer GPU
# attn_implementation = "flash_attention_2" # Use "flash_attention_2" when running on Ampere or newer GPU

# Hugging Face model id
model_id = "google/gemma-3-4b-it" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
new_model = f"{model_id.split('/')[1]}-bitext-cs-q8"
output_dir = f"{global_scratch}/{new_model}"
print(f"Output dir: \n\t{output_dir}")

bitext_ds = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# We are loading only the text stack
model_class = AutoModelForCausalLM

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation=attn_implementation,
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
    # max_memory={0: "16GB"}
)

# BitsAndBytesConfig: Enables 8-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id) # Load the Instruction Tokenizer to use the official Gemma template

# User prompt that combines the user query and the schema
system_prompt = "You are a top-rated customer service agent named John. Be polite to customers and answer all their questions."

def create_conversation(row):
  return {
    "messages": [{"role": "system", "content": system_prompt },
               {"role": "user", "content": row["instruction"]},
               {"role": "model", "content": row["response"]}]
  }

# Load dataset from the hub
dataset = load_dataset(bitext_ds, split="train")
# dataset = dataset.shuffle().select(range(500))
dataset = dataset.shuffle()

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False, num_proc = 28)
# split dataset into 20% test samples
dataset = dataset.train_test_split(test_size=0.2)

# Print formatted user prompt
print(dataset["train"][0])
print(type(dataset["train"][0]))


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

args = SFTConfig(
    output_dir=output_dir,                  # directory to save and repository id
    max_seq_length=512,                     # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=1,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    hub_model_id=new_model,                 # push model to hub with this id
    report_to="wandb",                      # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False,        # We template with special tokens
        "append_concat_token": True,        # Add EOS token as separator token between examples
    }
)

# Create Trainer object
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

print("Saving model")
trainer.save_model()

# free the memory again
del model
del trainer
torch.cuda.empty_cache()

# Merge Adapter and base model
print("Merging adapter and base model")
# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{output_dir}_adapter_fused", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained(f"{output_dir}_adapter_fused")

print("Pushing model to hub")
merged_model.push_to_hub(new_model)
processor.push_to_hub(new_model)
print("Pushed model to hub")

model_id = f"theflyingrahul/{new_model}"

print("Loading model from hub")
# Load Model with PEFT adapter
model = model_class.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch_dtype,
  attn_implementation=attn_implementation,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Test merged model
print("Testing FT model inference")
# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"]))
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)

print(f"Prompt:\n{prompt}")

# Generate
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=stop_token_ids, disable_compile=True)

print(f"Original Answer:\n{test_sample['messages'][2]['content']}")
# print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
print(f"Generated Answer:\n{outputs[0]['generated_text']}")