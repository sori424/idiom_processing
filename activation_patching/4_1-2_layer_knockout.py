from jaxtyping import Float
import transformer_lens.utils as utils
# Import stuff

# py39
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
from tqdm import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from functools import partial
import copy
import json
import os
import sys

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)
from neel_plotly import line, imshow, scatter
import transformer_lens.patching as patching

from collections import defaultdict
import argparse
from utils import *

def compute_global_means(model, knowns, replace, comp):

    global_means = {}
    sums = {layer: None for layer in range(model.cfg.n_layers)}
    counts = {layer: 0 for layer in range(model.cfg.n_layers)}
    
    for knowledge in tqdm(knowns, desc="Computing global MLP means"):
        original_string = knowledge['original_sentence']
        idiom = knowledge['idiom']
        tok_start, _ = find_token_range(original_string, idiom)

### for the literal/figurative string patching
        if replace == 'fig':
            clean_string = knowledge['figurative_sentence']
        elif replace == 'lit':
            clean_string = knowledge['literal_sentence']
        else:
            clean_string = knowledge['original_sentence']
        
        clean_tokens = model.to_tokens(clean_string, prepend_bos=False)
        tok_end = clean_tokens.tolist()[0].index(1606)

        _, clean_cache = model.run_with_cache(clean_tokens)
        for layer in range(model.cfg.n_layers):

            key = utils.get_act_name(f"{comp}_out", layer)
            mlp_activation = clean_cache[key]  # [batch, seq_len, hidden_dim]
            
            # Average activation over the idiom span for this example
            if replace == 'fig':
                if tok_start >= tok_end:
                    tok_start = tok_end - 1

            token_span_activation = mlp_activation[:, tok_start:tok_end, :]  # [batch, span_length, hidden_dim]
            if token_span_activation.size(1) == 0:
                print(f"Empty activation span for layer {layer} (tok_start: {tok_start}, tok_end: {tok_end}). Skipping example.", file=sys.stderr)
            token_activation_mean = token_span_activation.mean(dim=1)  # [batch, hidden_dim]
            if sums[layer] is None:
                sums[layer] = token_activation_mean.sum(dim=0)
            else:
                sums[layer] += token_activation_mean.sum(dim=0)
            counts[layer] += token_activation_mean.shape[0]
    
    for layer in range(model.cfg.n_layers):
        global_means[layer] = sums[layer] / counts[layer]
    return global_means


def patch_knockout_global_range(activation, hook, tok_start: int, tok_end: int, global_mean: torch.Tensor):

    activation_knocked = activation.clone()
    # Replace the entire idiom span with the global mean (broadcast to match span length)
    activation_knocked[:, tok_start:tok_end, :] = global_mean.unsqueeze(0).unsqueeze(1)
    return activation_knocked


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-3.2-1B", type=str)
    parser.add_argument('--replace', default="orig", type=str) #lit
    parser.add_argument('--component', default='mlp', type=str)

    args = parser.parse_args()

    device: torch.device = utils.get_device()

    model_name = args.model
    replace = args.replace
    comp = args.component

    model = HookedTransformer.from_pretrained(model_name)
    output_dir = f"./results/{model_name.split('/')[-1]}/knockout/{replace}"
    os.makedirs(output_dir, exist_ok=True)

    lit_dir = f"{output_dir}/{comp}/lit_drop"
    fig_dir = f"{output_dir}/{comp}/fig_drop"
    os.makedirs(lit_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    with open('/nethome/soyoung/idiom/idiom_process/data_gen/data/w_prefix_all_most_single_token_cand_literal_constrain.json', 'r') as f:
        knowns = json.load(f)

    # Precompute the global MLP means for each layer over the idiom spans
    global_means = compute_global_means(model, knowns, replace, comp)

    for i, knowledge in enumerate(tqdm(knowns)):
        original_string = knowledge['original_sentence']
        idiom = knowledge['idiom']
        id_list_, lit_list_ = list(knowledge['figurative_candidates'].keys()), list(knowledge['literal_candidates'].keys())
        id_list, lit_list = [int(x) for x in id_list_], [int(x) for x in lit_list_]

        literal_string = knowledge['literal_sentence']
        figurative_string = knowledge['figurative_sentence']

        if replace == 'lit':
            clean_string = literal_string
        elif replace == 'fig':
            clean_string = figurative_string
        else:
            clean_string = original_string

        clean_tokens = model.to_tokens(clean_string, prepend_bos=False)
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        
        tok_start, _ = find_token_range(original_string, idiom)
        tok_end = clean_tokens.tolist()[0].index(1606)

        patched_stream_diff_lit = torch.zeros(
            model.cfg.n_layers, 1, device=device, dtype=torch.float64
        ) 

        patched_stream_diff_fig = torch.zeros(
            model.cfg.n_layers, 1, device=device, dtype=torch.float64
        ) 

        
        for layer in range(model.cfg.n_layers):
            hook_fn = partial(patch_knockout_global_range,
                                    tok_start=tok_start,
                                    tok_end=tok_end,
                                    global_mean=global_means[layer]
                                )

            patched_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(utils.get_act_name(f"{comp}_out", layer), hook_fn)],
                return_type="logits",
            )

            patched_logit_diff_lit, patched_logit_diff_fig = prob_lse_diff(clean_logits, patched_logits, id_list, lit_list)
            patched_stream_diff_lit[layer] = patched_logit_diff_lit
            patched_stream_diff_fig[layer] = patched_logit_diff_fig


        patched_stream_diff_np_lit = patched_stream_diff_lit.cpu().numpy()
        np.savez(f'{lit_dir}/idiom_{i}.npz', patched_stream_diff_lit)

        patched_stream_diff_np_fig = patched_stream_diff_fig.cpu().numpy()
        np.savez(f'{fig_dir}/idiom_{i}.npz', patched_stream_diff_fig)