# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp

from tqdm import tqdm as tqdm
import torch
import copy
import numpy as np
from . import load_model_and_data as lmd

def ICL_inference_to_hidden_states(model, tokenizer, prompts): # [prompt] -> [layer][prompt][hidden_state]
    with torch.no_grad():
        ret = []
        hidden_states_in_layers = []
        for prompt in tqdm(prompts):
            torch.cuda.empty_cache()
            hidden_states_in_layer = []
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            for layer in range(len(result.hidden_states)):
                hidden_states_in_layer.append(result.hidden_states[layer][-1][-1].detach().to(torch.float).cpu().numpy())
            hidden_states_in_layers.append(hidden_states_in_layer)
        for layer in range(len(hidden_states_in_layers[0])):
            layer_hidden_states = []
            for prompt in hidden_states_in_layers:
                layer_hidden_states.append(prompt[layer])
            ret.append(layer_hidden_states)
        return ret

def ICL_inference_to_hidden_states_transposed(model, tokenizer, prompts): # [prompt] -> [prompt][layer][hidden_state]
    with torch.no_grad():
        hidden_states_in_layers = []
        for prompt in tqdm(prompts):
            torch.cuda.empty_cache()
            hidden_states_in_layer = []
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            for layer in range(len(result.hidden_states)):
                hidden_states_in_layer.append(result.hidden_states[layer][-1][-1].detach().to(torch.float).cpu().numpy())
            hidden_states_in_layers.append(hidden_states_in_layer)
        return hidden_states_in_layers

def encoder_inference_to_feature(model, tokenizer, queries):
    with torch.no_grad():
        representations = []
        for query in tqdm(queries[:512]):
            tknzd_data = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            representations.append(result.pooler_output[-1].detach().to(torch.float).cpu().numpy())
        return representations

def get_ppl(model, tokenizer, queries):
    with torch.no_grad():
        ret = []
        for query in tqdm(queries):
            tknzd_data = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, labels = tknzd_data)
            ret.append(result.loss.detach().to(torch.float).cpu().numpy().item())
        return ret
    
def ICL_inference_to_multi_token_hidden_states(model, tokenizer, prompts, tokens): # [prompt] -> [layer][prompt][token][hidden_state]
    with torch.no_grad():
        ret = []
        hidden_states_in_layers = []
        for prompt in tqdm(prompts):
            torch.cuda.empty_cache()
            hidden_states_in_layer = []
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True)
            for layer in range(len(result.hidden_states)):
                hidden_states_in_layer.append(result.hidden_states[layer][-1][tokens].detach().to(torch.float).cpu().numpy())
            hidden_states_in_layers.append(hidden_states_in_layer)
        for layer in range(len(hidden_states_in_layers[0])):
            layer_hidden_states = []
            for prompt in hidden_states_in_layers:
                layer_hidden_states.append(prompt[layer])
            ret.append(layer_hidden_states)

    real_ret = [[] for _ in range(len(tokens))]
    for i in range(len(tokens)):
        for layer in range(len(ret)):
            temp_layer = []
            for sample in ret[layer]:
                temp_layer.append(sample[i])
            real_ret[i].append(temp_layer)

    return real_ret

def ICL_inference_to_natural_hidden_states_and_attention(model, tokenizer, prompts): # [prompt] -> [layer][prompt][hidden_state]
    with torch.no_grad():
        ret_hidden = []
        ret_attention = []
        for prompt in tqdm(prompts):
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_hidden_layer = []
            res_attention_layer = []
            for layer in range(len(result.hidden_states)):
                res_hidden_layer.append(result.hidden_states[layer].detach().cpu())
            for layer in range(len(result.attentions)):
                res_attention_layer.append(result.attentions[layer].detach().cpu())
            ret_hidden.append(res_hidden_layer)
            ret_attention.append(res_attention_layer)
            del result
        return ret_hidden, ret_attention

def normlized_attention_score_for_single_sample(attention_score, hidden_state, query_key_values, w_s):
    ret = copy.deepcopy(attention_score)
    dimension = len(hidden_state[0][0])
    total_heads = len(attention_score[0][0])
    for layer in range(len(attention_score)):
        hidden_state_layer = hidden_state[layer][0]
        for head in range(len(attention_score[layer][0])):
            dimension_start = head * dimension // total_heads
            dimension_end = (head + 1) * dimension // total_heads
            for token in range(len(attention_score[layer][0][head])):
                v = query_key_values[layer](torch.tensor(hidden_state_layer[token]).to(torch.float).to(query_key_values[layer].device))[2//3*dimension][dimension_start:dimension_end].to(torch.float).cpu().numpy()
                v_before = np.array([0] * (dimension_start))
                v_after = np.array([0] * (dimension - dimension_end))
                v = np.concatenate([v_before, v, v_after])
                w = w_s[layer](torch.tensor(v).to(torch.float).to(w_s[layer].device))[0].to(torch.float).cpu().numpy()
                normw = np.linalg.norm(w)
                for qtoken in range(len(attention_score[layer][0][head])):
                    normlized_attention_score = attention_score[layer][0][head][qtoken][token] * normw
                    ret[layer][0][head][qtoken][token] = normlized_attention_score
    return ret

def copy_saliency_for_single_sample(hidden_states_numpy, model_modules):
    ret = []
    for layer in range(len(model_modules)):
        try:
            hidden_state = torch.tensor(hidden_states_numpy[layer]).to(model_modules[layer].device)
        except:
            hidden_state = torch.tensor(hidden_states_numpy[layer])
        hidden_state.requires_grad = True
        result = model_modules[layer](hidden_state)
        torch.sum(result[0][-1][-1]).backward()
        ret.append(hidden_state.grad.detach().cpu().numpy())
    return ret

def get_copy_magnitude_from_attention_for_single_sample(attention):
    ret = []
    for layer in range(len(attention)):
        ret.append(torch.mean(attention[layer][0], 0, keepdim = False))
    return ret

def get_copy_magnitude_from_attention_for_multi_sample(attention, Q, K):
    ret = [[] for _ in range(len(attention[0]))]
    for sample in attention:
        attention_magnitude = get_copy_magnitude_from_attention_for_single_sample(sample)
        for i in range(len(attention_magnitude)):
            ret[i].append(attention_magnitude[i][Q][K].detach().cpu().numpy().item())
    for i in range(len(ret)):
        ret[i] = np.mean(ret[i])
    return ret

def step3_get_fl_feature_and_lastftol_attention(model, tokenizer, prompts_with_label, experimentor, pythia = False): # ([sample][layer][tokens][hidden_state], [sample][layer][head][K])
    with torch.no_grad():
        ret_hidden = []
        ret_attention = []
        for prompt in tqdm(prompts_with_label):
            forerunner_loca, labels_loca = lmd.find_tokenized_label_word(tokenizer, experimentor, prompt, pythia)
            extract_loca = []
            for i in range(len(forerunner_loca)):
                extract_loca.append(forerunner_loca[i])
                extract_loca.append(labels_loca[i])
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_hidden_layer = []
            res_attention_layer = []
            for layer in range(len(result.hidden_states)):
                res_hidden_layer.append(result.hidden_states[layer][0][extract_loca].detach().cpu())
            for layer in range(len(result.attentions)):
                res_attention_head = []
                for head in range(len(result.attentions[layer][0])):
                    res_attention_head.append(result.attentions[layer][0][head][-2][extract_loca].detach().cpu())
                res_attention_layer.append(res_attention_head)
            ret_hidden.append(res_hidden_layer)
            ret_attention.append(res_attention_layer)
            del result
        return ret_hidden, ret_attention

def step2_get_fl_feature_and_lastftol_attention(model, tokenizer, prompts_with_label): # ([sample][layer][tokens][hidden_state], [sample][layer][head][K])
    with torch.no_grad():
        ret_hidden = []
        ret_attention = []
        for prompt in tqdm(prompts_with_label):
            forerunner_loca, labels_loca = [-2], [-1]
            extract_loca = []
            for i in range(len(forerunner_loca)):
                extract_loca.append(forerunner_loca[i])
                extract_loca.append(labels_loca[i])
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_hidden_layer = []
            res_attention_layer = []
            for layer in range(len(result.hidden_states)):
                res_hidden_layer.append(result.hidden_states[layer][0][extract_loca].detach().cpu())
            for layer in range(len(result.attentions)):
                res_attention_head = []
                for head in range(len(result.attentions[layer][0])):
                    res_attention_head.append(result.attentions[layer][0][head][-1][extract_loca].detach().cpu())
                res_attention_layer.append(res_attention_head)
            ret_hidden.append(res_hidden_layer)
            ret_attention.append(res_attention_layer)
            del result
        return ret_hidden, ret_attention
    
def step2_get_attention_of_different_location(model, tokenizer, prompts_with_label, experimentor, pythia = False): # ([sample][layer][tokens][hidden_state], [sample][layer][head][K])
    with torch.no_grad():
        normal_copy = []
        label_copy = []
        for prompt in tqdm(prompts_with_label):
            forerunner_loca, labels_loca = lmd.find_tokenized_label_word(tokenizer, experimentor, prompt, pythia)
            extract_loca = []
            for i in range(len(forerunner_loca)):
                extract_loca.append(labels_loca[i])
            torch.cuda.empty_cache()
            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            result = model(tknzd_data, output_hidden_states = True, output_attentions=True)
            res_attention_layer = []
            if len(normal_copy) == 0:
                for layer in range(len(result.attentions)):
                    normal_copy.append([])
                    label_copy.append([])
            for layer in range(len(result.attentions)):
                for head in range(len(result.attentions[layer][0])):
                    for index in range(labels_loca[-2] + 1, tknzd_data.shape[1]):
                        if index in extract_loca:
                            label_copy[layer].append(result.attentions[layer][0][head][index][index-1].detach().cpu().numpy().item() * tknzd_data.shape[1])
                        else:
                            normal_copy[layer].append(result.attentions[layer][0][head][index][index-1].detach().cpu().numpy().item() * tknzd_data.shape[1])
            del result
        return normal_copy, label_copy

def get_copy_magnitude_for_single_layer(ICL_attention, sample_index, layer):
    res = []
    for heads in ICL_attention[sample_index][layer]:
        res.append(heads[0].item())
    return res