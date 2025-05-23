# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp

from StaICC_Expe.StaICC.util import stable_random, functional
from tqdm import tqdm as tqdm

class hidden_calibration():
    # https://arxiv.org/abs/2406.16535
    def __init__(self, label_space) -> None:
        n_label = len(label_space)
        self.n_label = n_label
        self.centroid = []

    def train(
        self, 
        hidden_states_with_labels: list[list[float]]
    ):
        for list in hidden_states_with_labels:
            sum = [0] * len(list[0])
            for hidden_state in list:
                for i in range(len(hidden_state)):
                    sum[i] += hidden_state[i]
            self.centroid.append([x / len(list) for x in sum])
        print("Calibration Training Finished.\n")

    def inference(self, hidden_state) -> list[float]:
        L2_dist = [functional.L2_dist(hidden_state.tolist(), self.centroid[i]) for i in range(self.n_label)]
        normlized = [L2_dist[0] - L2_dist[i] for i in range(0, len(L2_dist))]
        return functional.softmax(normlized)
    
class layered_hidden_calibration():
    def __init__(self, label_space, layer_number, prompt_cut = "none", target_label_correction = True) -> None:
        self.label_space = label_space
        self.calibrations = []
        self.n_label = len(label_space)
        for i in range(layer_number):
            self.calibrations.append(hidden_calibration(label_space))
        self.prompt_cut = prompt_cut
        self.target_label_correction = target_label_correction
    
    def train(
        self, 
        default_prompt_maker: callable, # input: demos_lines: <list[(list[str], str)]>, query_line: <list[str]> return: prompt, recommendation: prompt_writter.write_prompt_from_dataline
        feedforward_with_layered_hidden_state: callable, # feedforward function, input: prompt: <str> return: hidden_state * layers
        calibration_set = None,
        calibration_number = 128,
        k = 4
    ):
        hidden_states = [[[] for _ in range(self.n_label)] for _ in range(len(self.calibrations))]
        my_random = stable_random.stable_random()
        demonstration_and_queue_samples = my_random.sample_index_set(calibration_number * (k + 1), len(calibration_set), allow_repetition=True)
        for i in range(calibration_number):
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((i + 1) / calibration_number * 100), 
                (i + 1), 
                calibration_number
            ), ">>" * int((i + 1) / calibration_number * 32), end="")
            demonstration_samples = demonstration_and_queue_samples[i * (k + 1) : (i + 1) * (k + 1) - 1]
            query_sample = demonstration_and_queue_samples[(i + 1) * (k + 1) - 1]
            query_label_index = calibration_set.find_index_from_label(calibration_set.get_label(demonstration_and_queue_samples[(i + 1) * (k + 1) - 1]))
            if not self.target_label_correction and self.prompt_cut == "label_words":
                query_label_index = (query_label_index + 1) % len(calibration_set._label_space)
            prompt = default_prompt_maker.write_prompt_from_dataline([calibration_set[demonstration_samples[j]] for j in range(k)], calibration_set[query_sample][0])
            cut_amount = -1
            if self.prompt_cut == "none":
                cut_amount = -1
            elif self.prompt_cut == "label_words":
                cut_amount = -1
                prompt = prompt + calibration_set._label_space[query_label_index] + ' '
            elif self.prompt_cut == "last_sentence_token":
                label_prefix_length = len(default_prompt_maker._label_prefix)
                cut_amount = -label_prefix_length - 1
            prompt = prompt[:cut_amount]

            hidden_state = feedforward_with_layered_hidden_state(prompts = [prompt])[0]
            for i in range(len(self.calibrations)):
                hidden_states[i][query_label_index].append(hidden_state[i])
        
        for i in range(len(self.calibrations)):
            self.calibrations[i].train(hidden_states[i])
    
    def single_layered_inference(self, layered_hidden_state_for_one_sample: list[list[float]]) -> list[list[float]]: # [layer][hidden_state] -> [layer][label_prob]
        ret = []
        for i in range(len(self.calibrations)):
            ret.append(self.calibrations[i].inference(layered_hidden_state_for_one_sample[i]))
        return ret

    def batched_layered_inference(self, layered_hidden_states: list[list[list[float]]]) -> list[list[list[float]]]: # [layer][sample][hidden_state] -> [layer][sample][label_prob]
        ret = [[] for _ in range(len(self.calibrations))]
        for sample_index in tqdm(range(len(layered_hidden_states[0]))):
            layered_hidden_state = []
            for i in range(len(self.calibrations)):
                layered_hidden_state.append(layered_hidden_states[i][sample_index])
            singleres = self.single_layered_inference(layered_hidden_state)
            for i in range(len(self.calibrations)):
                ret[i].append(singleres[i])
        return ret