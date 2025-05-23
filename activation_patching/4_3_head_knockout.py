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


def compute_global_means(model, knowns):

    global_means = {}
    device = utils.get_device()
    n_heads = model.cfg.n_heads
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model // n_heads

    sums = {
        layer: {head: torch.zeros(d_model, device=device) for head in range(n_heads)}
        for layer in range(n_layers)
    }
    counts = {
        layer: {head: 0 for head in range(n_heads)}
        for layer in range(n_layers)
    }
    
    for knowledge in tqdm(knowns, desc="Computing global Attn heads means"):
        clean_string = knowledge['original_sentence']
        idiom = knowledge['idiom']
        tok_start, _ = find_token_range(original_string, idiom)

        clean_tokens = model.to_tokens(clean_string, prepend_bos=False)
        tok_end = clean_tokens.tolist()[0].index(1606)

        _, clean_cache = model.run_with_cache(clean_tokens)
        for layer in range(model.cfg.n_layers):
            key = utils.get_act_name("z", layer, 'attn')
            attn_activation = clean_cache[key]
            for head in range(model.cfg.n_heads):
            
                token_span_activation = attn_activation[:, tok_start:tok_end, head, :]  # [batch, span_length, hidden_dim]
                if token_span_activation.size(1) == 0:
                    print(f"Empty activation span for layer {layer} (tok_start: {tok_start}, tok_end: {tok_end}). Skipping example.", file=sys.stderr)
                token_activation_mean = token_span_activation.mean(dim=1)  # [batch, hidden_dim]
                if sums[layer][head] is None:
                    sums[layer][head] = token_activation_mean.sum(dim=0)
                else:
                    sums[layer][head] += token_activation_mean.sum(dim=0)
                counts[layer][head] += token_activation_mean.shape[0]

    global_means = {}
    for layer in range(n_layers):
        global_means[layer] = {}
        for head in range(n_heads):
            cnt = counts[layer][head]
            global_means[layer][head] = sums[layer][head] / cnt
           
    return global_means


def patch_head_knockout_global_range(activation, hook, tok_start: int, tok_end: int, head_index: int, global_mean: torch.Tensor):

    activation_knocked = activation.clone()
    activation_knocked[:, tok_start:tok_end, head_index, :] = global_mean
    return activation_knocked


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-3.2-1B", type=str)
    args = parser.parse_args()

    device: torch.device = utils.get_device()

    model_name = args.model

    model = HookedTransformer.from_pretrained(model_name)
    output_dir = f"./results/{model_name.split('/')[-1]}/knockout/original"
    os.makedirs(output_dir, exist_ok=True)

    lit_dir = f"{output_dir}/head/lit_drop"
    fig_dir = f"{output_dir}/head/fig_drop"
    os.makedirs(lit_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)


    with open('/nethome/soyoung/idiom/idiom_process/data_gen/data/w_prefix_all_most_single_token_cand_literal_constrain.json', 'r') as f:
        knowns = json.load(f)

    # Precompute the global MLP means for each layer over the idiom spans
    global_means = compute_global_means(model, knowns)

    for i, knowledge in enumerate(tqdm(knowns)):
        original_string = knowledge['original_sentence']
        idiom = knowledge['idiom']
        id_list_, lit_list_ = list(knowledge['figurative_candidates'].keys()), list(knowledge['literal_candidates'].keys())
        id_list, lit_list = [int(x) for x in id_list_], [int(x) for x in lit_list_]

        clean_tokens = model.to_tokens(original_string, prepend_bos=False)
       
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        
        tok_start, _ = find_token_range(original_string, idiom)
        tok_end = clean_tokens.tolist()[0].index(1606)

        patched_head_z_diff_lit = torch.zeros(
            model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
        )
        
        patched_head_z_diff_fig = torch.zeros(
            model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
        )
            
        ## assign clean representation to hook, pos: where to insert
            # hook_fn = partial(patch_residual_component_replace, pos=tok_end, hook_pos= hook_pos, clean_cache=corrupt_cache)
        
        for layer in range(model.cfg.n_layers):
            for head_index in range(model.cfg.n_heads):
                hook_fn = partial(patch_head_knockout_global_range,
                                        tok_start=tok_start,
                                        tok_end=tok_end,
                                        head_index = head_index,
                                        global_mean=global_means[layer][head_index]
                                    )

                patched_head_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(utils.get_act_name("z", layer, "attn"), hook_fn)],
                    return_type="logits",
                )

                patched_logit_diff_lit, patched_logit_diff_fig = prob_lse_diff(clean_logits, patched_head_logits, id_list, lit_list)
                patched_head_z_diff_lit[layer, head_index] = patched_logit_diff_lit
                patched_head_z_diff_fig[layer, head_index] = patched_logit_diff_fig

        patched_heads_diff_np_lit = patched_head_z_diff_lit.cpu().numpy()
        patched_heads_diff_np_fig = patched_head_z_diff_fig.cpu().numpy()

        np.savez(f'{lit_dir}/idiom_{i}.npz', patched_heads_diff_np_lit)
        np.savez(f'{fig_dir}/idiom_{i}.npz', patched_heads_diff_np_fig)