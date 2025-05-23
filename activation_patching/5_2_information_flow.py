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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="meta-llama/Llama-3.2-1B", type=str)
    parser.add_argument('--replace', default="idiom_fig", type=str) # idiom_lit, idiomA_idiomB
    args = parser.parse_args()

    device: torch.device = utils.get_device()

    model_name = args.model
    replace = args.replace

    model = HookedTransformer.from_pretrained(model_name)
    output_dir = f"./results/{model_name.split('/')[-1]}/information_flow/{replace}"
    os.makedirs(output_dir, exist_ok=True)

    lit_resd_dir = f"{output_dir}/resid/lit_drop"
    fig_resd_dir = f"{output_dir}/resid/fig_drop"
    os.makedirs(output_dir + '/resid/lit_drop', exist_ok=True)
    os.makedirs(output_dir + '/resid/fig_drop', exist_ok=True)


    with open('/nethome/soyoung/idiom/idiom_process/data_gen/data/w_prefix_all_most_single_token_cand_literal_constrain.json', 'r') as f:
        knowns = json.load(f)

    indices = list(range(len(knowns)))
    paired_indices = get_derangement(indices)
    index_pairs = list(zip(indices, paired_indices))

    for i, j in tqdm(index_pairs):

        knowledge = knowns[i]
        clean_string = knowledge['original_sentence']

        if replace == 'idiom_lit':
            corrupt_string = knowledge['literal_sentence']
        elif replace == 'idiom_fig':
            corrupt_string = knowledge['figurative_sentence']
        elif replace == 'idiomA_idiomB':
            corrupt_string = knowns[j]['original_sentence']

        idiom = knowledge['idiom']
        id_list_, lit_list_ = list(knowledge['figurative_candidates'].keys()), list(knowledge['literal_candidates'].keys())
        id_list, lit_list = [int(i) for i in id_list_], [int(i) for i in lit_list_]

        corrupt_tokens = model.to_tokens(corrupt_string, prepend_bos=False)


    # first step: save the activation from clean string
        clean_tokens = model.to_tokens(clean_string, prepend_bos=False)
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupt_cache = model.run_with_cache(corrupt_tokens)

        pos = clean_tokens.tolist()[0].index(1606) # 1606 is the token id for ' because'
        hook_pos = corrupt_tokens.tolist()[0].index(1606)


    #### patching experiment
        patched_residual_stream_diff_lit = torch.zeros(
            model.cfg.n_layers, 1, device=device, dtype=torch.float64
        ) 

        patched_residual_stream_diff_fig = torch.zeros(
            model.cfg.n_layers, 1, device=device, dtype=torch.float64
        ) 


        for layer in range(model.cfg.n_layers):
            
            hook_fn = partial(patch_residual_component_replace, pos=pos, hook_pos=hook_pos, clean_cache=corrupt_cache)
        # RESDIDUAL
            patched_res_logits = model.run_with_hooks(
                clean_tokens,
                fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
                return_type="logits",
            )
            patched_res_logit_diff_lit, patched_res_logit_diff_fig = prob_lse_diff(clean_logits, patched_res_logits, id_list, lit_list)
            patched_residual_stream_diff_lit[layer] = patched_res_logit_diff_lit
            patched_residual_stream_diff_fig[layer] = patched_res_logit_diff_fig


        patched_residual_stream_diff_np_lit = patched_residual_stream_diff_lit.cpu().numpy()
        np.savez(f'{lit_resd_dir}/idiom_{i}.npz', patched_residual_stream_diff_np_lit)
        # print('Done with residual', file=sys.stderr)

        patched_residual_stream_diff_np_fig = patched_residual_stream_diff_fig.cpu().numpy()
        np.savez(f'{fig_resd_dir}/idiom_{i}.npz', patched_residual_stream_diff_np_fig)
