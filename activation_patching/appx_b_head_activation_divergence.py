import transformer_lens.utils as utils
from utils import *
# pip install transformer-lens
import torch
import itertools
from torch.nn.functional import normalize
from transformer_lens import HookedTransformer
from tqdm import tqdm
import torch.nn.functional as F
import json
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def load_model(name="gpt2-small", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(name, device=device)
    model.eval()
    return model

def get_head_vectors(model, cache, heads, positions):

    layer, head = heads
    tok_start, tok_end = positions
        # "pattern" is of shape (batch, n_heads, seq_len, seq_len)
    key = utils.get_act_name('z', layer, 'attn')
    attn_activation = cache[key]
    token_span_head_activation = attn_activation[:, tok_start:tok_end, head, :].mean(dim=1) 
    
    return token_span_head_activation
    

if __name__ == "__main__":

    model = load_model("meta-llama/Llama-3.2-1B")
    dis_list = [(0, 4), (0, 28), (1, 5), (2, 9), (0, 30), 
                (0, 8), (0, 19), (2, 2), (1, 25), (1, 18), (4, 31), (1, 31), (0, 12), 
                (0, 3), (2, 28), (2, 17), (2, 8), (2, 30), (1, 9), (4, 24)]
    
    sem_list = [(1, 3), (3, 28), (3, 30), (0, 10), (7, 16), (1, 20),
    (2, 4), (3, 12), (3, 14), (6, 27), (2, 19), (10, 11),
    (4, 26), (0, 18), (6, 31), (9, 22), (2, 5), (0, 9), (1,
    19), (0, 11)]

    rand_list = [(9, 20), (10, 30), (2, 11), (2, 23), (1, 1), (9, 23),
    (13, 24), (8, 31), (3, 8), (13, 9), (10, 1), (5, 15),
    (7, 14), (7, 24), (4, 5), (14, 2), (9, 17), (4, 0), (14,
    19), (6, 9)]

    filtered_indices = pickle.load(open("filtered_indices.pkl", "rb"))

    all_data = {}
    all_cos_sim = []
    all_orig_activations = []
    all_lit_activations = []

    with open('/nethome/soyoung/idiom/idiom_process/data_gen/data/w_prefix_all_most_single_token_cand_literal_constrain.json', 'r') as f:
        knowns = json.load(f)
    
    for i in range(3):
        if i == 0:
            target_list = dis_list
        elif i == 1:
            target_list = sem_list
        elif i == 2:
            target_list = rand_list

        for i, item in tqdm(enumerate(knowns)):
            cos_sims = []
            if i in filtered_indices:
                continue
            orig_string = item['original_sentence']
            lit_string = item['literal_sentence']

            orig_tokens = model.to_tokens(orig_string, prepend_bos=False)
            lit_tokens = model.to_tokens(lit_string, prepend_bos=False)    

            idiom = item['idiom']
            tok_start, _ = find_token_range(orig_string, idiom)
            orig_tok_end = orig_tokens.tolist()[0].index(1606)
            lit_tok_end = lit_tokens.tolist()[0].index(1606)


            with torch.no_grad():
                orig_logits, orig_cache = model.run_with_cache(orig_tokens)
                fig_logits, fig_cache = model.run_with_cache(lit_tokens)

            orig_activations = []
            lit_activations = []

        # change the bottom_cells : disamb top_cells: semantic

            for layer, head in target_list:
            # for head, layer in sampled_pairs[:20]:
                orig_vector = get_head_vectors(model, orig_cache, (layer, head), (tok_start, orig_tok_end))
                lit_vector = get_head_vectors(model, fig_cache, (layer, head), (tok_start, lit_tok_end))
                cos_sim = F.cosine_similarity(orig_vector, lit_vector, dim=1)

                orig_activations.append(orig_vector)
                lit_activations.append(lit_vector)
                cos_sims.append(cos_sim)

            all_orig_activations.append(orig_activations)
            all_lit_activations.append(lit_activations)
            all_cos_sim.append(cos_sims)

    all_data[i] = all_cos_sim

######## Visulization
    arr = np.stack([
    np.array([
        x.cpu().item() if isinstance(x, torch.Tensor) else float(x)
        for x in sublist
    ])
    for sublist in all_data[0]
    ])

    arr2 = np.stack([
        np.array([ 
            x.cpu().item() if isinstance(x, torch.Tensor) else float(x)
            for x in sublist
        ])
        for sublist in all_data[1]
    ])

    arr3 = np.stack([
        np.array([ 
            x.cpu().item() if isinstance(x, torch.Tensor) else float(x)
            for x in sublist
        ])
        for sublist in all_data[2]
    ])

    cmap = plt.get_cmap("tab10")
    color_order = [0, 1, 2, 3, 4, 5]
    
    means  = arr.mean(axis=1)
    means2 = arr2.mean(axis=1)
    means3 = arr3.mean(axis=1)

    # 1) choose your common bins
    nbins = 10
    counts1, bins = np.histogram(means,  bins=nbins)
    counts2, _    = np.histogram(means2, bins=bins)
    counts3, _    = np.histogram(means3, bins=bins)

    # 2) compute bar positions
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width       = (bins[1] - bins[0]) / 3

    # 3) plot grouped bars
    cmap = plt.get_cmap("tab10")
    color_order = [0, 1, 2, 3, 4, 5]
    plt.figure(figsize=(8,5))
    plt.bar(bin_centers - width, counts1, width=width,
            label='Idiomatic', color=cmap.colors[color_order[3]], alpha=0.7)
    plt.bar(bin_centers        , counts2, width=width,
            label='Semantic'    , color=cmap.colors[color_order[0]], alpha=0.7)
    plt.bar(bin_centers + width, counts3, width=width,
            label='Random'      , color='gray', alpha=0.7)

    # 4) styling
    plt.xlabel('Mean cosine similarity of top-20 attention heads', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(True, color='gray', linestyle='dotted', linewidth=0.5, alpha=0.4)

    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.rcParams["font.family"] = "Serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("cossim_grouped_bar.pdf", format='pdf', bbox_inches='tight')
    plt.show()
