# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp

from scipy import spatial
import numpy as np

def get_induction_correctness_for_single_layer(ICL_attention, experimentor, sample_index, layer):
    demo_indexs = experimentor.demonstration_sampler[sample_index]
    demo_labels = []
    for demo_index in demo_indexs:
        demo_labels.append(experimentor.get_label_space().index(experimentor.demonstration_set()[demo_index][1]))
    query_label = experimentor.get_label_space().index(experimentor.test_set()[sample_index][1])
    
    res = []
    for heads in ICL_attention[sample_index][layer]:
        temp = 0
        for i in range(len(demo_labels)):
            if demo_labels[i] == query_label:
                temp += heads[2 * i + 1]
            else:
                temp -= heads[2 * i + 1]
        res.append(temp.item())
    return res

def get_induction_magnitude_for_single_layer(ICL_attention, experimentor, sample_index, layer):
    demo_indexs = experimentor.demonstration_sampler[sample_index]
    demo_labels = []
    for demo_index in demo_indexs:
        demo_labels.append(experimentor.get_label_space().index(experimentor.demonstration_set()[demo_index][1]))
    query_label = experimentor.get_label_space().index(experimentor.test_set()[sample_index][1])
    
    res = []
    for heads in ICL_attention[sample_index][layer]:
        temp = 0
        for i in range(len(demo_labels)):
            temp += heads[2 * i + 1]
        res.append(temp.item())
    return res

def tokenized_length(tokenizer, prompt):
    tkized = tokenizer(prompt)['input_ids']
    return len(tkized)

def get_theresold_magnitude_from_prompt(tokenizer, prompt, induction_threthold_times, k):
    tkized = tokenizer(prompt)['input_ids']
    return induction_threthold_times * k / len(tkized)

def get_theresold_correctness_from_prompt(tokenizer, prompt, induction_threthold_times, label_space_length, k):
    return get_theresold_magnitude_from_prompt(tokenizer, prompt, induction_threthold_times, k) / label_space_length

def normalize(vector):
    _sum = sum(vector)
    return [x/_sum for x in vector]

def get_induction_likelihood_full_space_similarity(hidden_states, experimentor, sample_index, layer):
    demo_indexs = experimentor.demonstration_sampler[sample_index]
    demo_labels = []
    for demo_index in demo_indexs:
        demo_labels.append(experimentor.get_label_space().index(experimentor.demonstration_set()[demo_index][1]))
    query_label = experimentor.get_label_space().index(experimentor.test_set()[sample_index][1])
    
    res = [0] * len(experimentor.get_label_space())
    for i in range(len(demo_labels)):
        res[demo_labels[i]] += 1 - spatial.distance.cosine(hidden_states[sample_index][layer][2 * i], hidden_states[sample_index][layer][-2])
    res_sum = sum(res) + 1e-5
    res = [x/res_sum for x in res]
    return res[query_label]

def get_induction_likelihood_head(ICL_attention, experimentor, sample_index, layer):
    demo_indexs = experimentor.demonstration_sampler[sample_index]
    demo_labels = []
    for demo_index in demo_indexs:
        demo_labels.append(experimentor.get_label_space().index(experimentor.demonstration_set()[demo_index][1]))
    query_label = experimentor.get_label_space().index(experimentor.test_set()[sample_index][1])
    
    res = [[0] * len(experimentor.get_label_space()) for _ in range(len(ICL_attention[sample_index][layer]))]
    for heads in range(len(ICL_attention[sample_index][layer])):
        for i in range(len(demo_labels)):
            res[heads][demo_labels[i]] += ICL_attention[sample_index][layer][heads][2 * i + 1]
    res = np.array(res)
    for headline in range(len(res)):
        res[headline] = res[headline] / (sum(res[headline]) + 1e-5)
    ret = []
    for headline in res:
        ret.append(headline[query_label])
    return ret