import torch
from datasets import load_dataset
import random
from jaxtyping import Float
import numpy as np
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from scipy.stats import norm


def collect_embedding_mean(model):
    ds_name = "wikitext"
    raw_ds = load_dataset(
        ds_name,
        dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
    )

    raw_wiki = raw_ds['train']['text']
    raw_wiki = [item for item in raw_wiki if item != '']
    
    random.seed(42)
    raw_wiki = random.sample(raw_wiki, 300)
    alldata = []
    for i in raw_wiki:
        with torch.no_grad():
            outputs, cache = model.run_with_cache(i)
            embeddings = cache['hook_embed'][0]
            alldata.append(embeddings)
            # torch.cuda.empty_cache()
    alldata = torch.cat(alldata)
    noise_level = alldata.mean(dim=0)
    
    return noise_level


def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    head_index,
    pos,
    clean_cache,
):
    corrupted_head_vector[:, pos, head_index, :] = clean_cache[hook.name][
        :, pos, head_index, :
    ]
    return corrupted_head_vector

def patch_residual_component(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    clean_cache,
):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component

###### replacement

def patch_head_vector_replace(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    pos,
    hook_pos,
    head_index,
    clean_cache,
):
    # start_tok, end_tok = pos
    # patch_start_tok, patch_end_tok = hook_pos

    corrupted_head_vector[:, pos, head_index, :] = clean_cache[hook.name][:, hook_pos, head_index, :]
    return corrupted_head_vector

def patch_residual_component_replace(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    hook_pos,
    clean_cache,
):
    
    # start_tok, end_tok = pos
    # patch_start_tok, patch_end_tok = hook_pos
    
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, hook_pos, :]
    return corrupted_residual_component




def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def find_token_range(clean_string, substring):
    char_loc = clean_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    string_tokens = clean_string.split(' ')
    for i, t in enumerate(string_tokens):
        loc += len(t) + 1
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def corrupt_input(model, clean_string, tokens_to_mix, noise_level):

    rs = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    prng = lambda *shape: rs.randn(*shape)
    x = model.input_to_embed(clean_string, prepend_bos=False)

    noise_level = noise_level.view(1, 1, -1) 
    noise_fn = lambda x: noise_level * x

    if tokens_to_mix is not None:
        b, e = tokens_to_mix
        noise_shape = (x[0].shape[0], e-b, x[0].shape[2])
        # noise_shape = (x[0].shape[0], 1, x[0].shape[2]) # first and last token
        # noise_shape = (x[0].shape[0], e-b-2, x[0].shape[2]) # middle tokens
        noise_data = noise_fn(torch.from_numpy(prng(*noise_shape)).to(x[0].device))
        # Add noise to the specified token range
        x[0][:, b:e, :] = noise_data # all tokens
        # x[0][:, e, :] += noise_data.squeeze(1) # first and last token
        # x[0][:, b+1:e-1, :] += noise_data # middle tokens


    logits = model.unembed(x[0])
    predicted_token_ids = torch.argmax(logits, dim=-1)  # Shape: [1, seq_len]

    return predicted_token_ids


def get_derangement(lst):
    """
    Return a random derangement of lst (a permutation where no element remains in its original position).
    """
    while True:
        perm = lst[:]  # make a copy
        random.shuffle(perm)
        if all(perm[i] != lst[i] for i in range(len(lst))):
            return perm
        

def prob_lse_diff(original_logits, patched_logits, fig_answer_tokens, lit_answer_tokens):
    # Compute log-probs at the final time step
    orig_lp = torch.log_softmax(original_logits[0, -1, :], dim=-1)  # shape (V,)
    patch_lp = torch.log_softmax(patched_logits[0, -1, :], dim=-1)

    # Gather the log-probs for each candidate set
    fig_orig_lp = orig_lp[fig_answer_tokens]    # tensor of shape (|F|,)
    fig_patch_lp = patch_lp[fig_answer_tokens]
    lit_orig_lp = orig_lp[lit_answer_tokens]     # tensor of shape (|L|,)
    lit_patch_lp = patch_lp[lit_answer_tokens]

    # Log-sum-exp over each set
    S_fig_orig  = torch.logsumexp(fig_orig_lp,  dim=0)
    S_fig_patch = torch.logsumexp(fig_patch_lp, dim=0)
    S_lit_orig  = torch.logsumexp(lit_orig_lp,  dim=0)
    S_lit_patch = torch.logsumexp(lit_patch_lp, dim=0)

    # Return the deltas
    delta_lit = S_lit_patch  - S_lit_orig
    delta_fig = S_fig_patch  - S_fig_orig
    
    return delta_lit.item(), delta_fig.item()


def load_model(name="gpt2-small", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(name, device=device)
    model.eval()
    return model

def remove_outliers(data: list[float]) -> np.ndarray:
    arr = np.array(data)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return arr[(arr >= lower) & (arr <= upper)]

def remove_outliers_iqr(data_column, iqr_multiplier=1.0):
    q1 = np.percentile(data_column, 25)
    q3 = np.percentile(data_column, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    return [x for x in data_column if lower_bound <= x <= upper_bound]

def remove_outliers_zscore(data_column, z_threshold=1.0):
    data_column = np.array(data_column)
    mean_val = np.mean(data_column)
    std_val = np.std(data_column)
    
    z_scores = (data_column - mean_val) / std_val
    filtered_data = data_column[np.abs(z_scores) <= z_threshold]
    return filtered_data

def remove_outliers_percentile(data_column, lower_percentile=10, upper_percentile=90):
    lower_bound = np.percentile(data_column, lower_percentile)
    upper_bound = np.percentile(data_column, upper_percentile)
    
    
    return [x for x in data_column if lower_bound <= x <= upper_bound]


def compute_confidence_interval(data_by_layer, confidence=0.95, iqr_multiplier=1.0):
    means = []
    lowers = []
    uppers = []
    z = norm.ppf(1 - (1 - confidence) / 2)  # z = 1.96 for 95% CI

    for layer_data in zip(*data_by_layer):
        filtered = remove_outliers_iqr(layer_data)
        mean = np.mean(filtered)
        std = np.std(filtered, ddof=1)
        n = len(filtered)
        margin = z * (std / np.sqrt(n)) if n > 1 else 0
        means.append(mean)
        lowers.append(mean - margin)
        uppers.append(mean + margin)

    return means, lowers, uppers