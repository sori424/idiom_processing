# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp

import numpy as np
from tqdm import tqdm as tqdm

def sim_graph(features):
    simGraph = []
    for i in tqdm(range(len(features))):
        line = []
        for j in range(len(features)):
            if i == j:
                line.append(0)
            else:
                line.append(np.dot(features[i], features[j])/(np.linalg.norm(features[i]) * np.linalg.norm(features[j])))
        simGraph.append(line)
    return simGraph

def overlap(a, b):
    ret = 0
    for item in a:
        if item in b:
            ret += 1
    return ret

def kernel_alignment(simGraph_1, simGraph_2, k = 64):
    # kernel_alignment: calculate the alignment of two similarity graphs
    # simGraph_1: The first similarity graph
    # simGraph_2: The second similarity graph
    # k: The number of k-nearest neighbors
    # Return type: The mean, std, and a list of the individual alignment of each node
    aligns = []
    for i in range(len(simGraph_1)):
        aligns.append(
            overlap(np.argsort(simGraph_1[i])[::-1][:k], np.argsort(simGraph_2[i])[::-1][:k]) / k
        )
    return np.mean(aligns), np.std(aligns), aligns