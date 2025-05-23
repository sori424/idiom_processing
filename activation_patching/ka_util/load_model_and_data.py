# Official implementation of paper "Revisiting In-context Learning Inference Circuit in Large Language Models"
# Author: Hakaze Cho, yfzhao@jaist.ac.jp

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
import torch

def load_ICL_model(name: str, device: str = "cuda", huggingface_token = None, quantized = False, forcedownload = False, revision = None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if quantized else None
    if huggingface_token is not None:
        model = AutoModelForCausalLM.from_pretrained(name, token = huggingface_token, quantization_config = quantization_config, force_download = forcedownload, revision = revision)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config = quantization_config, force_download = forcedownload, revision = revision)
    if not quantized:
        model.to(device)
    model.eval()
    if huggingface_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(name, token = huggingface_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

def load_encode_model(name: str, device: str = "cuda", huggingface_token = None):
    if huggingface_token is not None:
        model = AutoModel.from_pretrained(name, token = huggingface_token)
    else:
        model = AutoModel.from_pretrained(name)
    model.to(device)
    model.eval()
    if huggingface_token is not None:
        tokenizer = AutoTokenizer.from_pretrained(name, token = huggingface_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

def load_data_from_StaICC_experimentor(experimentor, prompt_cut = "none", target_label_correction = True):
    _queries = experimentor.test_set()
    prompts = experimentor.prompt_set()[:len(_queries)]
    queries = []
    for i in range(len(_queries)):
        queries.append(_queries[i][0][0])
    cut_amount = -1
    if prompt_cut == "none":
        cut_amount = -1
    elif prompt_cut == "label_words":
        cut_amount = -1
        for i in range(len(prompts)):
            if target_label_correction:
                prompts[i] = prompts[i] + experimentor.prompt_former._label_space[_queries._label_space.index(_queries[i][1])] + ' '
            else:
                prompts[i] = prompts[i] + experimentor.prompt_former._label_space[(_queries._label_space.index(_queries[i][1]) + 1) % len(_queries._label_space)] + ' '
    elif prompt_cut == "last_sentence_token":
        label_prefix_length = len(experimentor.prompt_former._label_prefix)
        cut_amount = -label_prefix_length - 1
    
    for i in range(len(prompts)):
        prompts[i] = prompts[i][:cut_amount]
    return prompts, queries

def set_abstract_label_space(experimentor):
    new_label_space = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:len(experimentor.prompt_former._label_space)]
    experimentor.prompt_former.change_label_space(new_label_space)

def find_tokenized_label_word(tokenizer, experimentor, prompt, pythia = False):
    tokenized_prompt_input_ids = tokenizer(prompt)['input_ids']
    fore_runner_loca = []
    label_words_loca = []
    if pythia:
        divider = experimentor.prompt_former._label_prefix[:-1]
    else:
        divider = ' ' + experimentor.prompt_former._label_prefix[:-1]
    tokenized_divider = tokenizer(divider)['input_ids']
    tokenized_divider = tokenized_divider[-2:]
    for i in range(len(tokenized_prompt_input_ids)):
        if tokenized_prompt_input_ids[i:i + len(tokenized_divider)] == tokenized_divider:
            fore_runner_loca.append(i + 1)
            label_words_loca.append(i + 2)
    return fore_runner_loca, label_words_loca

def load_demonstrations_and_labels(experimentor):
    demo_inputs = []
    demo_labels = []
    for i in range(len(experimentor.demonstration_sampler)):
        temp_demo_inputs = []
        temp_demo_labels = []
        demonstration_indexs = experimentor.demonstration_sampler[i]
        for index in demonstration_indexs:
            temp_demo_inputs.append(experimentor.triplet_dataset.demonstration.get_input_text(index)[0])
            temp_demo_labels.append(experimentor.triplet_dataset.demonstration.get_label(index))
        demo_inputs.append(temp_demo_inputs)
        demo_labels.append(temp_demo_labels)

    queries = []
    for i in range(len(experimentor.test_set())):
        queries.append(experimentor.test_set().get_input_text(i)[0])
    
    return demo_inputs, demo_labels, queries