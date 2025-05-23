from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import json
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
import sys
from filterings import *
import numpy as np
import math
import builtins
import sys

model_name = 'meta-llama/Llama-3.3-70B-Instruct'
# model_name = "meta-llama/Llama-3.2-1B"


quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map='auto', 
    torch_dtype=torch.bfloat16, 
    quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cuda'


with open('../data/paraphrased.json', 'r') as f:
    data = json.load(f)


instruction = "Complete the following sentence with a single token. Do not provide any explanation."
system_prompt = ("You are a language model designed to follow precise instructions. "
                 "Your task is to complete the given sentence using exactly one token. "
                 "Do not include any explanation.")

all_orig = []
all_lit = []
all_fig = []

for entry in tqdm(data):
    original_sentence = entry['original_sentence']
    literal_sentence = entry['literal_sentence']
    figurative_sentence = entry['figurative_sentence']


    with torch.no_grad():

        original_input = tokenizer.encode(
            original_sentence,
            return_tensors="pt"
        ).to(device)

        literal_input = tokenizer.encode(
            literal_sentence,
            return_tensors="pt"
        ).to(device)

        figurative_input = tokenizer.encode(
            figurative_sentence,
            return_tensors="pt"
        ).to(device)


        # Run forward passes
        orig_generate = model(original_input, output_scores=True, return_dict=True)
        lit_generate = model(literal_input, output_scores=True, return_dict=True)
        fig_generate = model(figurative_input, output_scores=True, return_dict=True)

    # Assume batch size is 1; extract the first element from each output.
    orig_logits = orig_generate.logits[0, -1, :]  # shape: (vocab_size,)
    lit_logits = lit_generate.logits[0, -1, :]
    fig_logits = fig_generate.logits[0, -1, :]

    # Compute probabilities from logits
    orig_prob = torch.softmax(orig_logits, dim=-1)
    lit_prob = torch.softmax(lit_logits, dim=-1)
    fig_prob = torch.softmax(fig_logits, dim=-1)

    # Convert tensors to lists once rather than per-token conversion
    orig_logits_list = orig_logits.detach().cpu().tolist()
    orig_prob_list = orig_prob.detach().cpu().tolist()

    lit_logits_list = lit_logits.detach().cpu().tolist()
    lit_prob_list = lit_prob.detach().cpu().tolist()
    
    fig_logits_list = fig_logits.detach().cpu().tolist()
    fig_prob_list = fig_prob.detach().cpu().tolist()

    vocab_size = len(orig_logits_list)
    # Build dictionaries for only valid tokens
    orig_prob_dict = {token_id: {'logit': orig_logits_list[token_id], 'prob': orig_prob_list[token_id]}
                      for token_id in range(vocab_size)}
    lit_prob_dict = {token_id: {'logit': lit_logits_list[token_id], 'prob': lit_prob_list[token_id]}
                     for token_id in range(vocab_size)}
    fig_prob_dict = {token_id: {'logit': fig_logits_list[token_id], 'prob': fig_prob_list[token_id]}
                     for token_id in range(vocab_size)}

    all_orig.append(orig_prob_dict)
    all_lit.append(lit_prob_dict)
    all_fig.append(fig_prob_dict)


with open('lit_prob_dict_large.json', 'w', encoding="utf-8") as file:
    json.dump(all_lit, file, ensure_ascii=False, indent=4)

with open('id_prob_dict_large.json', 'w', encoding="utf-8") as file:
    json.dump(all_fig, file, ensure_ascii=False, indent=4)

with open('orig_prob_dict_large.json', 'w', encoding="utf-8") as file:
    json.dump(all_orig, file, ensure_ascii=False, indent=4)


def ratio(dict1, dict2):
    # Assume all keys in dictA appear in dictB
    keys = sorted(dict1.keys())
    A_values = np.array([dict1[k] for k in keys])
    B_values = np.array([dict2[k] for k in keys])
    
    # Compute the log difference for each key and return the mean
    log_diff = A_values - B_values
    return np.mean(log_diff)


lit_divergence = []
id_divergence = []
origc_lit_divergence = []
origc_fig_divergence = []

lit_dict = []
fig_dict = []
from nltk.corpus import stopwords

# Get English stopwords
stop_words = stopwords.words('english')

for i in range(len(all_orig)):

    orig_data = all_orig[i]
    orig_data_ = {key: np.log(value['prob']) for key, value in orig_data.items()}
    id_data = all_fig[i]
    id_data_ = {key: np.log(value['prob']) for key, value in id_data.items()}
    lit_data = all_lit[i]
    lit_data_ = {key: np.log(value['prob']) for key, value in lit_data.items()}
    
    diff_dict = {key: id_data_.get(key, 0) - lit_data_.get(key, 0) for key in set(lit_data_)}
    diff_abs_dict = {key: abs(id_data_.get(key, 0) - lit_data_.get(key, 0)) for key in set(lit_data_)}

    thres = sum(diff_abs_dict.values()) / len(diff_abs_dict)
    sub_filtered_abs_dict = {key: value for key, value in diff_abs_dict.items() if value >= thres}
    # print(len(sub_filtered_abs_dict))
    sub_filtered_dict = {key: diff_dict[key] for key, value in sub_filtered_abs_dict.items()}

    ### Top 20 tokens for each sentence
    lit_filtered_dict = {key: lit_data[key]['prob'] for key, value in sub_filtered_dict.items() if value < 0}
    id_filtered_dict = {key: id_data[key]['prob'] for key, value in sub_filtered_dict.items() if value > 0}

    lit_filtered_dict = {key: value for key, value in lit_filtered_dict.items() if tokenizer.decode(int(key)).islower() and tokenizer.decode(int(key)) not in [' ', " '"] and tokenizer.decode(int(key)).strip() not in stop_words}
    id_filtered_dict = {key: value for key, value in id_filtered_dict.items() if tokenizer.decode(int(key)).islower() and tokenizer.decode(int(key)) not in [' ', " '"] and tokenizer.decode(int(key)).strip() not in stop_words}

    litc_lit_filtered_dict = builtins.dict(sorted(lit_filtered_dict.items(), key=lambda item: item[1], reverse=True)[:20])
    idc_id_filtered_dict = builtins.dict(sorted(id_filtered_dict.items(), key=lambda item: item[1], reverse=True)[:20])

    lit_dict.append({int(key): {'token':tokenizer.decode(int(key)), 'logit': lit_data[key]['logit'], 'prob':lit_data[key]['prob']} for key, value in litc_lit_filtered_dict.items()})
    fig_dict.append({int(key): {'token':tokenizer.decode(int(key)), 'logit': id_data[key]['logit'], 'prob':id_data[key]['prob']} for key, value in idc_id_filtered_dict.items()})

    # top 20 tokens in different context
    origc_id_filtered_dict = {key: orig_data_[key] for key, value in idc_id_filtered_dict.items()}
    origc_lit_filtered_dict = {key: orig_data_[key] for key, value in litc_lit_filtered_dict.items()}

    litc_id_filterd_dict = {key: lit_data_[key] for key, value in idc_id_filtered_dict.items()}
    idc_lit_filtered_dict = {key: id_data_[key] for key, value in litc_lit_filtered_dict.items()}

    origc_lit_div = ratio(litc_lit_filtered_dict, origc_lit_filtered_dict)
    origc_fig_div = ratio(idc_id_filtered_dict, origc_id_filtered_dict)

    lit_div = ratio(litc_lit_filtered_dict, idc_lit_filtered_dict)
    id_div = ratio(idc_id_filtered_dict, litc_id_filterd_dict)

    origc_lit_divergence.append(origc_lit_div)
    origc_fig_divergence.append(origc_fig_div)
    lit_divergence.append(lit_div)
    id_divergence.append(id_div)

origc_lit_divergence = [x for x in origc_lit_divergence if not math.isnan(x)]
origc_fig_divergence = [x for x in origc_fig_divergence if not math.isnan(x)]
lit_divergence = [x for x in lit_divergence if not math.isnan(x)]
id_divergence = [x for x in id_divergence if not math.isnan(x)]

print(sum(origc_lit_divergence) / len(origc_lit_divergence), file=sys.stderr) # literal context vs. original context with literal candidates
print(sum(origc_fig_divergence) / len(origc_fig_divergence) , file=sys.stderr) # figurative context vs. original context with figurative candidates
print("Average JS-divergence for literal tokens: ", sum(lit_divergence) / len(lit_divergence),  file=sys.stderr)
print("Average JS-divergence for figurative tokens: ", sum(id_divergence) / len(id_divergence),  file=sys.stderr)



all_dict = []
for i, item in enumerate(data):
    dict = {}
    idiom = item['idiom']
    orig = item['original_sentence']
    lit = item['literal_sentence']
    fig = item['figurative_sentence']

    dict['idiom'] = idiom
    dict['original_sentence'] = orig
    dict['literal_sentence'] = lit
    dict['figurative_sentence'] = fig
    
    for candidate in lit_dict[i].values():
        candidate['prob'] = f"{candidate['prob']:.20f}"
    dict['literal_candidates'] = lit_dict[i]

    for candidate in fig_dict[i].values():
        candidate['prob'] = f"{candidate['prob']:.20f}" 

    dict['figurative_candidates'] = fig_dict[i]

    all_dict.append(dict)    

# Write the list of dictionaries to a JSON file
with open("single_token_candidates.json", "w") as f:
    json.dump(all_dict, f, indent=4)  # 'indent' makes the file easier to read
# with open("fig_most_single_token_dict.json", "w") as f: