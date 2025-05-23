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

def patch_attention_score(
    corrupted_residual_component: Float[torch.Tensor, "batch head query_pos key_pos"],
    hook,
    bc_pos,
    idiom_span,
    clean_cache,
    path
):
    
    start_tok, end_tok = idiom_span
    
    if path == 'idiom_bc':
        corrupted_residual_component[:, :, bc_pos, start_tok:end_tok] = float('-inf') # path interrupted when idiom to because
    elif path == 'bc_last':
        corrupted_residual_component[:, :, -1, bc_pos] = float('-inf') # path interrupted when because to last token
    elif path == 'idiom_last':
        corrupted_residual_component[:, :, -1, start_tok:end_tok] = float('-inf') # path from idiom to last
    return corrupted_residual_component

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-3.2-1B", type=str)
    parser.add_argument('--path', default="idiom_bc", type=str) # idiom_bc, bc_last, idiom_last
    args = parser.parse_args()

    device: torch.device = utils.get_device()

    model_name = args.model
    path = args.path

    model = HookedTransformer.from_pretrained(model_name)
    output_dir = f"./results/{model_name.split('/')[-1]}/attention_edge_patch/{path}"
    os.makedirs(output_dir, exist_ok=True)

    lit_resd_dir = f"{output_dir}/lit_drop"
    fig_resd_dir = f"{output_dir}/fig_drop"
    os.makedirs(output_dir + '/attn/lit_drop', exist_ok=True)
    os.makedirs(output_dir + '/attn/fig_drop', exist_ok=True)

    with open('/nethome/soyoung/idiom/idiom_process/data_gen/data/w_prefix_all_most_single_token_cand_literal_constrain.json', 'r') as f:
        knowns = json.load(f)

    for i, knowledge in enumerate(tqdm(knowns)):

        clean_string = knowledge['original_sentence']
        idiom = knowledge['idiom']
        id_list_, lit_list_ = list(knowledge['figurative_candidates'].keys()), list(knowledge['literal_candidates'].keys())
        id_list, lit_list = [int(i) for i in id_list_], [int(i) for i in lit_list_]

    # first step: save the activation from clean string
        clean_tokens = model.to_tokens(clean_string , prepend_bos=False)
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)

        tok_start, _ = find_token_range(clean_string, idiom) # tok_end is for the subsequent token index
        bc_pos = clean_tokens.tolist()[0].index(1606) # 1606 is the token id for ' because'

    #### patching experiment
        
        patched_attn_diff_lit = torch.zeros(
            model.cfg.n_layers, 1, device=device, dtype=torch.float32
        )

        patched_attn_diff_fig = torch.zeros(
            model.cfg.n_layers, 1, device=device, dtype=torch.float32
        )


        for layer in range(model.cfg.n_layers):
            
        ## assign clean representation to hook, pos: where to insert
            # hook_fn = partial(patch_residual_component_replace, pos=tok_end, hook_pos= hook_pos, clean_cache=corrupt_cache)
            hook_fn = partial(patch_attention_score, bc_pos=bc_pos, idiom_span=(tok_start, bc_pos), clean_cache=clean_cache, path=path)


        # ATTENTION
            patched_attn_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(utils.get_act_name("attn_scores", layer), hook_fn)],
                return_type="logits",
            )

            patched_attn_logit_diff_lit,  patched_attn_logit_diff_fig = prob_lse_diff(clean_logits, patched_attn_logits, id_list, lit_list)
            patched_attn_diff_lit[layer] = patched_attn_logit_diff_lit
            patched_attn_diff_fig[layer] = patched_attn_logit_diff_fig


        patched_attn_diff_np_lit = patched_attn_diff_lit.cpu().numpy()
        np.savez(f'{lit_resd_dir}/idiom_{i}.npz', patched_attn_diff_np_lit)

        patched_attn_diff_np_fig = patched_attn_diff_fig.cpu().numpy()
        np.savez(f'{fig_resd_dir}/idiom_{i}.npz', patched_attn_diff_np_fig)
