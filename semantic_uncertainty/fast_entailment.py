from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
import wandb
import os, shutil
import pickle
import sys

model_type = sys.argv[1]
temp = sys.argv[2]
wandb_id = sys.argv[3]

run = wandb.init(
    project='fast-entailment', 
    job_type="inference", 
    anonymous="allow"
)

api = wandb.Api()

def restore(name, wandb_runid, filename):
    run = api.run(f'theflyingrahul-indian-institute-of-management-bangalore/semantic_uncertainty/{wandb_runid}')
    # The pkl files are stored in an internal nested directory structure. We need to dynamically determine the internal dir structure and load the pkl file
    # Try to find and download the file
    for file in run.files():
        if file.name.endswith(filename):
            file.download(replace=True, exist_ok=False)
            print(f"Downloaded: {file.name}")
            # Move it to pwd
            new_location = os.path.basename(file.name)  # just the filename
            shutil.move(file.name, f"{os.getcwd()}/{filename.replace('.', '_')}/{name}.pkl")
            break
    else:
        print(f"File {filename} not found")

generated_run_ids = {
    f"{model_type}_t{temp}": wandb_id,
}

compute_run_ids = {
    f"{model_type}_llama_t{temp}": wandb_id,
}

filename = 'uncertainty_measures.pkl'
for key, value in compute_run_ids.items():
    restore(key, value, filename)

# Load pkl file in the directory into a dictionary
uncertainty_measures_pkl = {}
directory = 'uncertainty_measures_pkl'
for filename in os.listdir(directory):
    if filename.startswith(f"{model_type}_llama_t{temp}") and filename.endswith('.pkl'):
        label = filename[:-4]  # Remove the '.pkl' extension
        with open(os.path.join(directory, filename), 'rb') as file:
            uncertainty_measures_pkl[label] = pickle.load(file)

print(uncertainty_measures_pkl.keys())

filename = 'validation_generations.pkl'
for key, value in generated_run_ids.items():
    restore(key, value, filename)

# Load pkl file in the directory into a dictionary
generated_data = dict()
for filename in os.listdir('validation_generations_pkl'):
    if filename.startswith(f"{model_type}_t{temp}") and filename.endswith('.pkl'):
        label = filename[:-4]  # Remove the '.pkl' extension
        with open(os.path.join('validation_generations_pkl', filename), 'rb') as file:
            generated_data[label] = pickle.load(file)

print(generated_data.keys())

# No random sampling, we work with all 400 labels
label_indices = {label: list(generated_data[list(generated_run_ids.keys())[0]].keys()).index(label) for label in generated_data[list(generated_run_ids.keys())[0]].keys()}
# print(label_indices)
captured_info = {}

for config, data in generated_data.items():
    captured_info[config] = {}
    for label in data.keys():
        captured_info[config][label] = {
            "prompt": data[label]['question'],
            "truth": data[label]['reference']['answers']['text'],
            "generated": [response[0] for response in data[label]['responses']]
        }

# print(captured_info)
print(len(label_indices))

# fetch semantic classes from uncertainty_measures pkl files
# There's a problem here: we don't know the mapping of the labels to the semantic classes. uncertainty_measures pkl files have no order to the labels. Were they sorted?

# NOTE: Python v>3.7 preserves dictionary order. We just need to get the index of the randomly sampled labels from the generated_data dict.

# uncertainty_measures_pkl keys: {modelname}_{entailmodel}_t{0.1}
for config, data in captured_info.items():
    modelname = config.split('_')[0]
    temp = config.split('_')[1].lstrip('t')
    for label in data.keys():
        # count number of unique semantic classes and add to the captured_info dict
        llama_semantic_ids = uncertainty_measures_pkl[f"{modelname}_llama_t{temp}"]["semantic_ids"][label_indices[label]]
        captured_info[config][label]['llama_semantic_class_count'] = len(set(llama_semantic_ids))

#####################################################################################

model_name = "meta-llama/Llama-3.2-3B-Instruct"

kwargs = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)}

tokenizer = AutoTokenizer.from_pretrained(
    model_name, device_map="auto",
    token_type_ids=None
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto",
    attn_implementation="flash_attention_2", # disable flash attention (for GPUs older than Ampere)
    **kwargs
)

def equivalence_prompt(text1, text2, question):

    prompt = f"""We are evaluating responses from two customer service chatbots to a customer's question: `{question}`\n"""
    prompt += "Here are two possible answers:\n"
    prompt += f"Possible Answer 1: `{text1}`\nPossible Answer 2: `{text2}`\n"
    prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with `entailment`, `contradiction`, or `neutral`.\n"""
    prompt += "Response:"""

    return prompt

def call_llm(prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)

    # Generate predictions
    outputs = model.generate(
        inputs["input_ids"],
        # max_length=200,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generated_entails_truth(config, label):
    print(f"Processing label: {label}")
    print("Sequence:" , end=" ")
    generated_entails_truth = []
    for i in range(len(captured_info[config][label]['generated'])):
        print (i, end=" ")
        prompt = equivalence_prompt(captured_info[config][label]['generated'][i], captured_info[config][label]['truth'], captured_info[config][label]['prompt'])
        predicted_text = call_llm(prompt)
        judgement = predicted_text[len(prompt):].split()[0]
        # print(f"Judgement: {judgement}")
        if "entailment" in judgement.lower():
            generated_entails_truth.append(1)
        elif "contradiction" in judgement.lower():
            generated_entails_truth.append(-1)
            print(f"\tContradiction in label: {label}")
        elif "neutral" in judgement.lower():
            generated_entails_truth.append(0)
        else:
            print(f"\tUnknown judgement in label: {label}")
            generated_entails_truth.append(-2)
        captured_info[config][label]['generated_entails_truth'] = generated_entails_truth
        print(judgement, end = " -> ")
    print()

configurations = generated_run_ids.keys()
print("Testing if generated responses entail truth")
for config in configurations:
    print(f"Processing configuration: {config}")
    label_count = 0
    for label in captured_info[config].keys():
        print(f"Label sequence: {label_count}", end=" ")
        generated_entails_truth(config, label)
        label_count += 1

del model
import torch
import gc
gc.collect()
torch.cuda.empty_cache()

# cosines
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set device to GPU 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('all-mpnet-base-v2', device=device)

def get_cosine_similarity(sentence1, sentence2):
    # Encode sentences to get their embeddings
    embeddings = model.encode([sentence1, sentence2], device=device)
    
    # Compute cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return similarity

get_cosine_similarity("This is a test sentence.", "This is another test sentence.")

def compute_cosine_similarities(config, label):
    print(f"Processing label: {label}")
    print("Sequence:" , end=" ")
    generated_cosine_with_truth = []
    for i in range(len(captured_info[config][label]['generated'])):
        print (i, end=" ")
        cosine = get_cosine_similarity(captured_info[config][label]['generated'][i], captured_info[config][label]['truth'])
        generated_cosine_with_truth.append(cosine)
        print(cosine, end = " -> ")
    captured_info[config][label]['generated_cosine_with_truth'] = generated_cosine_with_truth
    print()

configurations = list(generated_run_ids.keys())

print("Computing semantic cosine similarity between generated and truth")
for config in configurations:
    print(f"Processing configuration: {config}")
    label_count = 0
    for label in captured_info[config].keys():
        print(f"Label sequence: {label_count}", end=" ")
        compute_cosine_similarities(config, label)
        label_count += 1

del model

output_filename = f"{configurations[0]}_entail_cosines.pkl"

# Write the captured_info dictionary to the pkl file
with open(output_filename, "wb") as file:
    pickle.dump(captured_info, file)

print(f"captured_info has been written to {output_filename}")