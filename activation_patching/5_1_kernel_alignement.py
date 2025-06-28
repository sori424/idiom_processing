from ka_util import load_model_and_data, kernel_alignment, inference
import StaICC
import matplotlib.pyplot as plt
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

## Some definations for the plots.
plt.style.use('default')
plt.rc('font',family='Cambria Math')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Cambria Math'] + plt.rcParams['font.serif']

# Cell 2: Model and huggingfacetoken configurations

## The huggingface model name to be tested as the LM for ICL. 
## Recommended: "meta-llama/Meta-Llama-3-8B", "EleutherAI/pythia-6.9b", "tiiuae/falcon-7b", "meta-llama/Meta-Llama-3-70B", "tiiuae/falcon-40b"
ICL_model_name = "meta-llama/Llama-3.2-1B" 

## Whether to use the quantized version of the model. 
## Recommended: Keep it default.
quantized = False if ICL_model_name in ["meta-llama/Meta-Llama-3-8B", "EleutherAI/pythia-6.9b", "tiiuae/falcon-7b"] else True

## The huggingface model name to be tested as the reference encoder.
## Recommended: "BAAI/bge-m3"
encoder_model_name = "BAAI/bge-m3"

## The huggingface token to access the model. If you use the Llama model, you need to set this.
huggingface_token = "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"


# Experiment parameters

## The selected token type to calculate the KA. Alternative: "none" (forerunner token s), "label_words" (y), "last_sentence_token" (x).
ICL_selected_token_type = "last_sentence_token" 

## The demonstration numbers. Recommended: 0, 1, 2, 4, 8, 12.
k = 0

## The used dataset index from the StaICC library. Alternative: 0, 1, 2, 3, 4, 5. See the README.md for more information.
dataset_index = 2 

## The used dataset index from the StaICC library for the controlled experiment. Fixed to TEE (index 6).
pesudo_dataset_index = 7  

## Force the ICL_model to reload, even the ICL_model is already in the variables. 
## Recommended: False.
model_forced_reload = False


# Cell 4: Load the models.

vars_dict = vars() if "ICL_model" in vars() else locals()
# if "ICL_model" not in vars_dict or model_forced_reload:
ICL_model, ICL_tknz = load_model_and_data.load_ICL_model(ICL_model_name, huggingface_token = huggingface_token, quantized = False)
encoder_model, encoder_tknz = load_model_and_data.load_encode_model(encoder_model_name, huggingface_token = huggingface_token)


with open('/nethome/soyoung/idiom/idiom_process/data_gen/data/w_prefix_all_most_single_token_cand_literal_constrain.json', 'r') as f:
    data = json.load(f)

filtered_indices = pickle.load(open("/nethome/soyoung/idiom/idiom_process/activation_patch/filtered_indices.pkl", "rb"))

fig_prompts = [] # paraphrased sentences
lit_prompts = []


last_idiom_token_queries = [] # idiom_sentence
bc_token_queries = []
second_subsequent_token_queries = []
last_token_queries = []


for i, item in enumerate(data):
    if i in filtered_indices:
        continue
    orig_sentence_last_token = item['original_sentence']
    last_token_queries.append(orig_sentence_last_token)

    orig_sentence_second_subsequent_token = item['original_sentence'].split(' because')[0] + ' because ' + item['original_sentence'].split(' because')[1].split(' ')[1]
    second_subsequent_token_queries.append(orig_sentence_second_subsequent_token)

    orig_sentence_last_idiom_token = item['original_sentence'].split(' because')[0] 
    last_idiom_token_queries.append(orig_sentence_last_idiom_token)

    orig_sentence_bc_token = item['original_sentence'].split(' because')[0] + ' because'
    bc_token_queries.append(orig_sentence_bc_token)
    
    fig_sentence = item['figurative_sentence'].split(' because')[0]
    if 'would' in fig_sentence:
        try:
            fig_prompts.append(fig_sentence.split('would ')[1])
        except:
            fig_prompts.append(fig_sentence.split("wouldn't ")[1])
    else:
        fig_prompts.append(' '.join(fig_sentence.split(' ')[1:]))
   
    lit_sentence = item['literal_sentence'].split(' because')[0]
    if 'would' in lit_sentence:
        lit_prompts.append(lit_sentence.split('would ')[1])
    else:
        lit_prompts.append(' '.join(lit_sentence.split(' ')[1:]))


# Cell 5: Get the ICL hidden states and the encoder features, also the pesudo encoder features from another dataset defined by the pesudo_dataset_index.

ICL_hidden_states_last_idiom_token = inference.ICL_inference_to_hidden_states(ICL_model, ICL_tknz, last_idiom_token_queries)
ICL_hidden_states_bc_token = inference.ICL_inference_to_hidden_states(ICL_model, ICL_tknz, bc_token_queries)
ICL_hidden_states_bc_subsequent_token = inference.ICL_inference_to_hidden_states(ICL_model, ICL_tknz, second_subsequent_token_queries)
ICL_hidden_states_last_token = inference.ICL_inference_to_hidden_states(ICL_model, ICL_tknz, last_token_queries)

fig_encoder_feature = inference.encoder_inference_to_feature(encoder_model, encoder_tknz, fig_prompts)
lit_encoder_feature = inference.encoder_inference_to_feature(encoder_model, encoder_tknz, lit_prompts)

# Cell 6: Calculate the similarity map and the kernel alignment. Refer to `util/kernel_alignment.py` for more details.

## Calculate the similarity map (defined as $\delta: \mathcal{X}\rightarrow\mathbb{H}^{d}$ in the Appendix A.2).
ICL_sim_map_last_idiom = []
ICL_sim_map_bc = []
ICL_sim_map_bc_subsequent = []
ICL_sim_map_last = []

for i, layer_hidden_state in enumerate(ICL_hidden_states_last_idiom_token):
    ICL_sim_map_last_idiom.append(kernel_alignment.sim_graph(layer_hidden_state))
    ICL_sim_map_bc.append(kernel_alignment.sim_graph(ICL_hidden_states_bc_token[i]))
    ICL_sim_map_bc_subsequent.append(kernel_alignment.sim_graph(ICL_hidden_states_bc_subsequent_token[i]))
    ICL_sim_map_last.append(kernel_alignment.sim_graph(ICL_hidden_states_last_token[i]))
    

fig_encoder_sim_map = kernel_alignment.sim_graph(fig_encoder_feature)
lit_encoder_sim_map = kernel_alignment.sim_graph(lit_encoder_feature)


## Calculate the kernel alignment.
### The organization of the results: res_kernel_alignment[layer_index]: (mean, std, individual_values)
fig_res_kernel_alignment_last_idiom = []
fig_res_kernel_alignment_bc = []
fig_res_kernel_alignment_bc_subsequent = []
fig_res_kernel_alignment_last = []

lit_res_kernel_alignment_last_idiom = []
lit_res_kernel_alignment_bc = []
lit_res_kernel_alignment_bc_subsequent = []
lit_res_kernel_alignment_last = []


for i, layer_sim_graph in enumerate(ICL_sim_map_last_idiom):
    fig_res_kernel_alignment_last_idiom.append(kernel_alignment.kernel_alignment(layer_sim_graph, fig_encoder_sim_map))
    fig_res_kernel_alignment_bc.append(kernel_alignment.kernel_alignment(ICL_sim_map_bc[i], fig_encoder_sim_map))
    fig_res_kernel_alignment_bc_subsequent.append(kernel_alignment.kernel_alignment(ICL_sim_map_bc_subsequent[i], fig_encoder_sim_map))
    fig_res_kernel_alignment_last.append(kernel_alignment.kernel_alignment(ICL_sim_map_last[i], fig_encoder_sim_map))    

    lit_res_kernel_alignment_last_idiom.append(kernel_alignment.kernel_alignment(layer_sim_graph, lit_encoder_sim_map))
    lit_res_kernel_alignment_bc.append(kernel_alignment.kernel_alignment(ICL_sim_map_bc[i], lit_encoder_sim_map))
    lit_res_kernel_alignment_bc_subsequent.append(kernel_alignment.kernel_alignment(ICL_sim_map_bc_subsequent[i], lit_encoder_sim_map))
    lit_res_kernel_alignment_last.append(kernel_alignment.kernel_alignment(ICL_sim_map_last[i], lit_encoder_sim_map))

# Cell 7: Data preview.

fig_last_idiom_avg_kernel_alignment_for_plot = []
fig_bc_avg_kernel_alignment_for_plot = []
fig_bc_subsequent_avg_kernel_alignment_for_plot = []
fig_last_token_avg_kernel_alignment_for_plot = []

lit_last_idiom_avg_kernel_alignment_for_plot = []
lit_bc_avg_kernel_alignment_for_plot = []
lit_bc_subsequent_avg_kernel_alignment_for_plot = []
lit_last_token_avg_kernel_alignment_for_plot = []

for i, line in enumerate(fig_res_kernel_alignment_last_idiom[1:]):
    fig_last_idiom_avg_kernel_alignment_for_plot.append(line[0])
    fig_bc_avg_kernel_alignment_for_plot.append(fig_res_kernel_alignment_bc[i][0])
    fig_bc_subsequent_avg_kernel_alignment_for_plot.append(fig_res_kernel_alignment_bc_subsequent[i][0])
    fig_last_token_avg_kernel_alignment_for_plot.append(fig_res_kernel_alignment_last[i][0])

    lit_last_idiom_avg_kernel_alignment_for_plot.append(lit_res_kernel_alignment_last_idiom[i][0])
    lit_bc_avg_kernel_alignment_for_plot.append(lit_res_kernel_alignment_bc[i][0])
    lit_bc_subsequent_avg_kernel_alignment_for_plot.append(lit_res_kernel_alignment_bc_subsequent[i][0])
    lit_last_token_avg_kernel_alignment_for_plot.append(lit_res_kernel_alignment_last[i][0])


x = np.arange(len(fig_last_idiom_avg_kernel_alignment_for_plot))
plt.figure(figsize=(5, 5))
plt.xlabel("Layers", fontsize = 17)
plt.ylabel("Kernel Alignment", fontsize = 17)
plt.rcParams["font.family"] = "Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
color_order = [0, 1, 2, 4, 5, 3]

cmap = plt.get_cmap("tab10")

plt.plot(fig_last_idiom_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_i, s_f)$', color=cmap.colors[color_order[0]], linewidth=2)
plt.plot(lit_last_idiom_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_i, s_l)$', color=cmap.colors[color_order[0]], linewidth=2, linestyle='dotted')

plt.plot(fig_bc_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_{i+1}, s_f)$', color=cmap.colors[color_order[1]], linewidth=2)
plt.plot(lit_bc_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_{i+1}, s_l)$', color=cmap.colors[color_order[1]], linewidth=2, linestyle='dotted')

plt.plot(fig_bc_subsequent_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_{i+2}, s_f)$', color=cmap.colors[color_order[2]], linewidth=2)
plt.plot(lit_bc_subsequent_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_{i+2}, s_l)$', color=cmap.colors[color_order[2]], linewidth=2, linestyle='dotted')

plt.plot(fig_last_token_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_{-1}, s_f)$', color=cmap.colors[color_order[3]], linewidth=2)
plt.plot(lit_last_token_avg_kernel_alignment_for_plot, label=r'$(\mathrm{x}_{-1}, s_l)$', color=cmap.colors[color_order[3]], linewidth=2, linestyle='dotted')

plt.legend(loc="lower right", borderaxespad=1, fontsize=25)
plt.grid(True, color='gray', linestyle='dotted', linewidth=0.5, alpha=0.4)
plt.xticks(fontsize=17)
plt.yticks(np.arange(0.5, 0.75, 0.05), fontsize=17)
# Save the figure as a PDF file.
plt.savefig("kernel_alignment_plot.pdf", format='pdf', bbox_inches='tight')
plt.show()
