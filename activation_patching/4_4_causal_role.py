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
import pickle

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)
from neel_plotly import line, imshow, scatter
import transformer_lens.patching as patching

from collections import defaultdict
import argparse
from utils import *

def patch_residual(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    hook_pos,
    clean_cache,
    magnitude
):
    
    start_tok, end_tok = pos
    patch_start_tok, patch_end_tok = hook_pos
    
    corrupted_residual_component[:, start_tok:end_tok, :] = magnitude * clean_cache[hook.name][:, patch_start_tok:patch_end_tok, :]
    return corrupted_residual_component


def patch_head(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    pos,
    hook_pos,
    head_index,
    clean_cache,
    magnitude
):
    # clean_cache[hook.name]
    start_idx, end_idx = pos
    patch_start_tok, patch_end_tok = hook_pos

    corrupted_head_vector[:, start_idx:end_idx, head_index, :] = magnitude * clean_cache[hook.name][
        :, patch_start_tok:patch_end_tok, head_index, :] 
        
    return corrupted_head_vector


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-3.2-1B", type=str)
    parser.add_argument('--target', default="idiom", type=str) #lit
    
    args = parser.parse_args()

    model_name = args.model
    target = args.target

    model = load_model(model_name)

    with open('/nethome/soyoung/idiom/idiom_process/data_gen/data/w_prefix_all_most_single_token_cand_literal_constrain.json', 'r') as f:
        knowns = json.load(f)

    filtered_indices = pickle.load(open("filtered_indices.pkl", "rb"))

    if target == 'idiom':
        target_mlps = [1,2]
        target_heads = [(0, 4), (0, 28), (1, 5), (2, 9), (0, 30), (0, 8), (0, 19),
                        (2, 2), (1, 25), (1, 18), (4, 31), (1, 31), (0, 12), (0,3), 
                        (2, 28), (2, 17), (2, 8), (2, 30), (1, 9), (4, 24)]
    elif target == 'sem':
        target_heads = [(1, 3), (3, 28), (3, 30), (0, 10), (7, 16), (1, 20), (2, 4), 
                        (3, 12), (3, 14), (6, 27), (2, 19), (10, 11), (4, 26), (0, 18), 
                        (6, 31), (9, 22), (2, 5), (0, 9), (1, 19), (0, 11)]
        target_mlps = [10,11]
    else:
        target_heads = [(9, 20), (10, 30), (2, 11), (2, 23), (1, 1), (9, 23),
                        (13, 24), (8, 31), (3, 8), (13, 9), (10, 1), (5, 15), (7, 14), 
                        (7, 24), (4, 5), (14, 2), (9, 17), (4, 0), (14, 19), (6, 9)]


    magnitude = [1.0]
    all_magnitude = {m: {'lit': [], 'fig': []} for m in magnitude}

    for mag in magnitude:
        all_fig = []
        all_lit = []
        for i, item in tqdm(enumerate(knowns)):
            
            if i not in filtered_indices:

                orig_string = item['original_sentence']
                orig_tokens = model.to_tokens(orig_string, prepend_bos=False)

                idiom = item['idiom']
                tok_start, _ = find_token_range(orig_string, idiom)
                tok_end = orig_tokens.tolist()[0].index(1606)

                literal_string = item['literal_sentence']
                # literal_string = item['figurative_sentence']
                literal_tokens = model.to_tokens(literal_string, prepend_bos=False)
                # lit_tok_start = tok_start
                lit_tok_end = literal_tokens.tolist()[0].index(1606)

                id_list_, lit_list_ = list(item['figurative_candidates'].keys()), list(item['literal_candidates'].keys())
                id_list, lit_list = [int(i) for i in id_list_], [int(i) for i in lit_list_]


                with torch.no_grad():
                    orig_logits, orig_cache = model.run_with_cache(orig_tokens)
                    lit_logits, lit_cache = model.run_with_cache(literal_tokens)

                fwd_hooks = []

                pos = (tok_start, lit_tok_end)
                hook_pos = (tok_start, tok_end)

    ## change targets
                for layer, head in target_heads:
                    hook_fn = partial(patch_head, pos=pos, hook_pos=hook_pos, head_index=head, clean_cache=orig_cache, magnitude=mag)
                    fwd_hooks.append((utils.get_act_name("z", layer, "attn"), hook_fn))

                for layer in target_mlps:
                    hook_fn = partial(patch_residual, pos=pos, hook_pos=hook_pos, clean_cache=orig_cache, magnitude=mag)
                    fwd_hooks.append((utils.get_act_name("mlp_out", layer), hook_fn))

                try:
                    patched_logits = model.run_with_hooks(
                        literal_tokens,
                        fwd_hooks=fwd_hooks,
                        return_type="logits",
                    )
                except:
                    # print(i)
                    pass
                lit, fig = prob_lse_diff(lit_logits, patched_logits, id_list, lit_list)
                all_lit.append(lit)
                all_fig.append(fig)
        
        all_magnitude[mag]['lit'] = all_lit
        all_magnitude[mag]['fig'] = all_fig 

    
    all_lit_vals = remove_outliers(all_magnitude[1]['lit'])
    all_fig_vals = remove_outliers(all_magnitude[1]['fig'])
    print(f'=== Steering with {target} components ===')
    print(r'$\Delta F(s_a)$', all_fig_vals.mean())
    print(r'$\Delta L(s_a)$', all_lit_vals.mean())

