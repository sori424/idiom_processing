from tqdm import tqdm as tqdm
from typing import Optional, Tuple, Union
import random
import types
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)

import math

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

def make_falcon_model(model):
    model.transformer.layered_mask_inference_for_falcon = types.MethodType(layered_mask_inference_for_falcon, model.transformer)
    model.transformer._update_causal_mask = types.MethodType(_update_causal_mask, model.transformer)
    model.layered_mask_inference_for_falcon_CLM = types.MethodType(layered_mask_inference_for_falcon_CLM, model)

def make_llama_model(model):
    model.model.layered_mask_inference_for_llama = types.MethodType(layered_mask_inference_for_llama, model.model)
    model.layered_mask_inference_for_llama_CLM = types.MethodType(layered_mask_inference_for_llama_CLM, model)

def make_llama_model_Q(model):
    model.model.layered_mask_inference_for_llama = types.MethodType(layered_mask_inference_for_llamaQ, model.model)
    model.layered_mask_inference_for_llama_CLM = types.MethodType(layered_mask_inference_for_llama_CLM, model)



def find_tokenized_label_word(tokenizer, experimentor, prompt):
    tokenized_prompt_input_ids = tokenizer(prompt)['input_ids']
    fore_runner_loca = []
    label_words_loca = []
    divider = ' ' + experimentor.prompt_former._label_prefix[:-1]
    tokenized_divider = tokenizer(divider)['input_ids']
    tokenized_divider = tokenized_divider[-2:]
    for i in range(len(tokenized_prompt_input_ids)):
        if tokenized_prompt_input_ids[i:i + len(tokenized_divider)] == tokenized_divider:
            fore_runner_loca.append(i + 1)
            label_words_loca.append(i + 2)
    return fore_runner_loca, label_words_loca

class AttentionMaskForSingleHead:
    @staticmethod
    def default_causal_attention_mask(length):
        return torch.tril(torch.ones(length, length), diagonal = 0), 0

    @staticmethod
    def single_attention_mask_from_forerunner_to_forerunner(length, label_words_loca):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        count = 0
        for i in label_words_loca:
            for j in label_words_loca:
                if i >= j:
                    basic_mask[i - 1][j - 1] = 0
                    count += 1
        return basic_mask, count

    @staticmethod
    def single_attention_mask_for_query_FRT(length, query_start):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        count = 0
        for i in range(query_start, length):
            basic_mask[-1][i] = 0
            count += 1
        return basic_mask, count
    
    @staticmethod
    def attention_mask_for_demo_FRT_to_LW(length, label_words_loca):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        count = 0
        for i in label_words_loca[:-1]:
            basic_mask[i][i - 1] = 0
            count += 1
        return basic_mask, count
    
    @staticmethod
    def attention_mask_for_demo_FRT_to_all_LW(length, label_words_loca):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        count = 0
        for i in label_words_loca[:-1]:
            for j in label_words_loca[:-1]:
                if i >= j:
                    basic_mask[i][j - 1] = 0
                    count += 1
        return basic_mask, count
    
    @staticmethod
    def attention_mask_for_demo_text_to_demo_FRT(length, label_words_loca):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        anchors = []
        starts = [0]
        for i in label_words_loca:
            anchors.append(i - 1)
        for i in range(len(anchors) - 1):
            starts.append(label_words_loca[i] + 1)
        count = 0
        for i in range(len(anchors)):
            for j in range(starts[i], anchors[i]):
                basic_mask[anchors[i]][j] = 0
                count += 1
        return basic_mask, count

    @staticmethod
    def attention_mask_for_query_FRT_to_LW(length, label_words_loca):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        count = 0
        for i in label_words_loca[:-1]:
            basic_mask[-1][i] = 0
            count += 1
        return basic_mask, count

    @staticmethod
    def random_mask(length, amount):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        count = 0
        while count < amount:
            i = random.randint(0, length - 1)
            j = random.randint(0, length - 1)
            if basic_mask[i][j] == 1:
                basic_mask[i][j] = 0
                count += 1
        return basic_mask, count

    @staticmethod
    def full_attention_mask(length):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        count = 0
        for i in range(length):
            for j in range(length):
                basic_mask[i][j] = 0
                count += 1
        return basic_mask, count
    

class AttentionMaskForLayer:
    @staticmethod
    def default_causal_attention_mask(length, head_number):
        mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]
    
    @staticmethod
    def full_mask_for_heads(length, head_number, masked_heads):
        basic_mask = AttentionMaskForSingleHead.default_causal_attention_mask(length)[0]
        full_mask, one_count = AttentionMaskForSingleHead.full_attention_mask(length)
        mask = []
        for i in range(head_number):
            if i in masked_heads:
                mask.append(full_mask)
            else:
                mask.append(basic_mask)
        mask = torch.stack(mask, dim=0)
        return mask, 0

    @staticmethod
    def attention_mask_from_forerunner_to_forerunner(length, label_words_loca, head_number):
        mask = AttentionMaskForSingleHead.single_attention_mask_from_forerunner_to_forerunner(length, label_words_loca)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]

    @staticmethod
    def single_attention_mask_for_query_FRT(length, query_start, head_number):
        mask = AttentionMaskForSingleHead.single_attention_mask_for_query_FRT(length, query_start)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]
    
    @staticmethod
    def attention_mask_for_demo_FRT_to_LW(length, label_words_loca, head_number):
        mask = AttentionMaskForSingleHead.attention_mask_for_demo_FRT_to_LW(length, label_words_loca)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]
    
    @staticmethod
    def attention_mask_for_demo_FRT_to_all_LW(length, label_words_loca, head_number):
        mask = AttentionMaskForSingleHead.attention_mask_for_demo_FRT_to_all_LW(length, label_words_loca)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]
    
    @staticmethod
    def attention_mask_for_demo_text_to_demo_FRT(length, label_words_loca, head_number):
        mask = AttentionMaskForSingleHead.attention_mask_for_demo_text_to_demo_FRT(length, label_words_loca)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]

    @staticmethod
    def attention_mask_for_query_FRT_to_LW(length, label_words_loca, head_number):
        mask = AttentionMaskForSingleHead.attention_mask_for_query_FRT_to_LW(length, label_words_loca)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]

    @staticmethod
    def random_mask(length, amount, head_number):
        mask = AttentionMaskForSingleHead.random_mask(length, amount)
        return mask[0].unsqueeze(dim=0).repeat(head_number, 1, 1), mask[1]


class AttentionMaskForModel:
    @staticmethod
    def convert_mask_to_attention_mask(mask, dtype = torch.float32):
        if dtype == bool:
            return mask.to(dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    @staticmethod
    def default_causal_attention_mask(length, head_number, layer_number, dtype = torch.float32):
        mask = AttentionMaskForLayer.default_causal_attention_mask(length, head_number)
        return [AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype) * layer_number], [mask[1] * layer_number]
    
    @staticmethod
    def full_mask_for_heads(length, head_number, masked_heads_in_layer, layer_number, dtype = torch.float32):
        mask = []
        for i in range(layer_number):
            mask.append(AttentionMaskForModel.convert_mask_to_attention_mask(AttentionMaskForLayer.full_mask_for_heads(length, head_number, masked_heads_in_layer[i])[0].unsqueeze(dim=0), dtype))
        return mask, 0
    
    @staticmethod
    def attention_mask_from_forerunner_to_forerunner(length, label_words_loca, head_number, layer_number, mask_layer_start, mask_layer_end, dtype = torch.float32):
        masks = []
        counts = []
        for i in range(layer_number):
            if i < mask_layer_start or i >= mask_layer_end:
                mask = AttentionMaskForLayer.default_causal_attention_mask(length, head_number)
            else:
                mask = AttentionMaskForLayer.attention_mask_from_forerunner_to_forerunner(length, label_words_loca, head_number)
            masks.append(AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype))
            counts.append(mask[1])
        return masks, counts

    @staticmethod
    def attention_mask_for_query_FRT_to_LW(length, label_words_loca, head_number, layer_number, mask_layer_start, mask_layer_end, dtype = torch.float32):
        masks = []
        counts = []
        for i in range(layer_number):
            if i < mask_layer_start or i >= mask_layer_end:
                mask = AttentionMaskForLayer.default_causal_attention_mask(length, head_number)
            else:
                mask = AttentionMaskForLayer.attention_mask_for_query_FRT_to_LW(length, label_words_loca, head_number)
            masks.append(AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype))
            counts.append(mask[1])
        return masks, counts

    @staticmethod
    def attention_mask_for_demo_FRT_to_LW(length, label_words_loca, head_number, layer_number, mask_layer_start, mask_layer_end, dtype = torch.float32):
        masks = []
        counts = []
        for i in range(layer_number):
            if i < mask_layer_start or i >= mask_layer_end:
                mask = AttentionMaskForLayer.default_causal_attention_mask(length, head_number)
            else:
                mask = AttentionMaskForLayer.attention_mask_for_demo_FRT_to_LW(length, label_words_loca, head_number)
            masks.append(AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype))
            counts.append(mask[1])
        return masks, counts
    
    @staticmethod
    def attention_mask_for_demo_FRT_to_all_LW(length, label_words_loca, head_number, layer_number, mask_layer_start, mask_layer_end, dtype = torch.float32):
        masks = []
        counts = []
        for i in range(layer_number):
            if i < mask_layer_start or i >= mask_layer_end:
                mask = AttentionMaskForLayer.default_causal_attention_mask(length, head_number)
            else:
                mask = AttentionMaskForLayer.attention_mask_for_demo_FRT_to_all_LW(length, label_words_loca, head_number)
            masks.append(AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype))
            counts.append(mask[1])
        return masks, counts
    
    @staticmethod
    def attention_mask_for_demo_text_to_demo_FRT(length, label_words_loca, head_number, layer_number, mask_layer_start, mask_layer_end, dtype = torch.float32):
        masks = []
        counts = []
        for i in range(layer_number):
            if i < mask_layer_start or i >= mask_layer_end:
                mask = AttentionMaskForLayer.default_causal_attention_mask(length, head_number)
            else:
                mask = AttentionMaskForLayer.attention_mask_for_demo_text_to_demo_FRT(length, label_words_loca, head_number)
            masks.append(AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype))
            counts.append(mask[1])
        return masks, counts

    @staticmethod
    def single_attention_mask_for_query_FRT(length, query_start, head_number, layer_number, mask_layer_start, mask_layer_end, dtype = torch.float32):
        masks = []
        counts = []
        for i in range(layer_number):
            if i < mask_layer_start or i >= mask_layer_end:
                mask = AttentionMaskForLayer.default_causal_attention_mask(length, head_number)
            else:
                mask = AttentionMaskForLayer.single_attention_mask_for_query_FRT(length, query_start, head_number)
            masks.append(AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype))
            counts.append(mask[1])
        return masks, counts
    
    @staticmethod
    def random_mask(length, amount: list[int], head_number, layer_number, dtype = torch.float32):
        masks = []
        for i in range(layer_number):
            mask = AttentionMaskForLayer.random_mask(length, amount[i], head_number)
            masks.append(AttentionMaskForModel.convert_mask_to_attention_mask(mask[0].unsqueeze(dim=0), dtype))
        return masks, amount

def head_ablation(
    prompts,
    model,
    tokenizer,
    experimentor,
    inference_type = "llama",
    ablation_head_list = None,
    cache_empty: callable = torch.cuda.empty_cache(), # GPU cache empty function. Can be torch.cuda.empty_cache.
    dtype = None
):
    if dtype is None:
        dtype = model.dtype
    label_space = experimentor.get_label_space()
    ret_experiment = []
    ret_control = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            plain_tokenized = tokenizer(prompt)['input_ids']
            if cache_empty is not None:
                cache_empty()
            if ablation_head_list is None:
                attention_mask = None
            else:
                attention_mask = AttentionMaskForModel.full_mask_for_heads(
                    length = len(plain_tokenized),
                    head_number = model.config.num_attention_heads,
                    masked_heads_in_layer = ablation_head_list,
                    layer_number = model.config.num_hidden_layers,
                    dtype=dtype
                )[0]

            if inference_type == "falcon":
                inference_func = model.layered_mask_inference_for_falcon_CLM
            elif inference_type == "llama":
                inference_func = model.layered_mask_inference_for_llama_CLM

            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) # flexable??
            result = inference_func(tknzd_data, attention_mask = attention_mask)
            full_vocab_prob = result['logits'][0][-1].detach().to(torch.float).cpu().numpy()
            tokenized_label_space = [tokenizer(label).input_ids[-1] for label in label_space]
            label_space_logits = [full_vocab_prob[token] for token in tokenized_label_space]
            label_space_prob = torch.nn.functional.softmax(torch.tensor(label_space_logits))
            del tknzd_data
            del result
            ret_experiment.append(label_space_prob.tolist())
    return ret_experiment


def Masked_ICL_inference(
    prompts,
    model,
    tokenizer,
    experimentor,
    inference_type = "llama",
    mask_start_layer: int = 0,
    mask_end_layer: int = 1,
    attention_mask_type = "default_causal_attention_mask", # default_causal_attention_mask, xq_to_sq, LW_to_query_FRT, demo_FRT_to_LW, demo_text_to_demo_FRT, demo_FRT_to_LW_star, all_F_to_F
    cache_empty: callable = torch.cuda.empty_cache(), # GPU cache empty function. Can be torch.cuda.empty_cache.
    dtype = None, # Data type for the mask. If None, it will be the same as the model's dtype.
    run_control_experiment_parallelly = True # Whether to run control experiment parallelly.
):
    if dtype is None:
        dtype = model.dtype
    label_space = experimentor.get_label_space()
    ret_experiment = []
    ret_control = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            plain_tokenized = tokenizer(prompt)['input_ids']
            if cache_empty is not None:
                cache_empty()
            if attention_mask_type == "default_causal_attention_mask":
                attention_mask = [None, 0]
            elif attention_mask_type == "xq_to_sq": # Line 3
                fore_runner_loca, label_words_loca = find_tokenized_label_word(tokenizer, experimentor, prompt)
                attention_mask = AttentionMaskForModel.single_attention_mask_for_query_FRT(
                    length = len(plain_tokenized),
                    query_start = label_words_loca[-2] + 1,
                    head_number = model.config.num_attention_heads,
                    layer_number = model.config.num_hidden_layers,
                    mask_layer_start = mask_start_layer,
                    mask_layer_end = mask_end_layer,
                    dtype=dtype
                )
            elif attention_mask_type == "yi_to_sq": # Line 5
                fore_runner_loca, label_words_loca = find_tokenized_label_word(tokenizer, experimentor, prompt)
                attention_mask = AttentionMaskForModel.attention_mask_for_query_FRT_to_LW(
                    length = len(plain_tokenized),
                    label_words_loca = label_words_loca,
                    head_number = model.config.num_attention_heads,
                    layer_number = model.config.num_hidden_layers,
                    mask_layer_start = mask_start_layer,
                    mask_layer_end = mask_end_layer,
                    dtype=dtype
                )
            elif attention_mask_type == "si_to_yi": # Line 4
                fore_runner_loca, label_words_loca = find_tokenized_label_word(tokenizer, experimentor, prompt)
                attention_mask = AttentionMaskForModel.attention_mask_for_demo_FRT_to_LW(
                    length = len(plain_tokenized),
                    label_words_loca = label_words_loca,
                    head_number = model.config.num_attention_heads,
                    layer_number = model.config.num_hidden_layers,
                    mask_layer_start = mask_start_layer,
                    mask_layer_end = mask_end_layer,
                    dtype=dtype
                )
            elif attention_mask_type == "xi_to_si": # Line 2
                fore_runner_loca, label_words_loca = find_tokenized_label_word(tokenizer, experimentor, prompt)
                attention_mask = AttentionMaskForModel.attention_mask_for_demo_text_to_demo_FRT(
                    length = len(plain_tokenized),
                    label_words_loca = label_words_loca,
                    head_number = model.config.num_attention_heads,
                    layer_number = model.config.num_hidden_layers,
                    mask_layer_start = mask_start_layer,
                    mask_layer_end = mask_end_layer,
                    dtype=dtype
                )
            elif attention_mask_type == "si_to_y[:i+1]": # Unused
                fore_runner_loca, label_words_loca = find_tokenized_label_word(tokenizer, experimentor, prompt)
                attention_mask = AttentionMaskForModel.attention_mask_for_demo_FRT_to_all_LW(
                    length = len(plain_tokenized),
                    label_words_loca = label_words_loca,
                    head_number = model.config.num_attention_heads,
                    layer_number = model.config.num_hidden_layers,
                    mask_layer_start = mask_start_layer,
                    mask_layer_end = mask_end_layer,
                    dtype=dtype
                )
            elif attention_mask_type == "si_to_si": # Table 2 
                fore_runner_loca, label_words_loca = find_tokenized_label_word(tokenizer, experimentor, prompt)
                attention_mask = AttentionMaskForModel.attention_mask_from_forerunner_to_forerunner(
                    length = len(plain_tokenized),
                    label_words_loca = label_words_loca,
                    head_number = model.config.num_attention_heads,
                    layer_number = model.config.num_hidden_layers,
                    mask_layer_start = mask_start_layer,
                    mask_layer_end = mask_end_layer,
                    dtype=dtype
                )

            attention_mask, masked_count = attention_mask[0], attention_mask[1]

            if run_control_experiment_parallelly:
                controlled_attention_mask = AttentionMaskForModel.random_mask(
                    length = len(plain_tokenized),
                    amount = masked_count,
                    head_number = model.config.num_attention_heads,
                    layer_number = model.config.num_hidden_layers,
                    dtype=dtype
                )[0]

            if inference_type == "falcon":
                inference_func = model.layered_mask_inference_for_falcon_CLM
            elif inference_type == "llama":
                inference_func = model.layered_mask_inference_for_llama_CLM

            tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) # flexable??
            result = inference_func(tknzd_data, attention_mask = attention_mask)
            if run_control_experiment_parallelly:
                result_control = inference_func(tknzd_data, attention_mask = controlled_attention_mask)
            full_vocab_prob = result['logits'][0][-1].detach().to(torch.float).cpu().numpy()
            if run_control_experiment_parallelly:
                full_vocab_prob_control = result_control['logits'][0][-1].detach().to(torch.float).cpu().numpy()
            tokenized_label_space = [tokenizer(label).input_ids[-1] for label in label_space]
            label_space_logits = [full_vocab_prob[token] for token in tokenized_label_space]
            if run_control_experiment_parallelly:
                label_space_logits_control = [full_vocab_prob_control[token] for token in tokenized_label_space]
            label_space_prob = torch.nn.functional.softmax(torch.tensor(label_space_logits))
            if run_control_experiment_parallelly:
                label_space_prob_control = torch.nn.functional.softmax(torch.tensor(label_space_logits_control))
            del tknzd_data
            del result
            if run_control_experiment_parallelly:
                del result_control
            ret_experiment.append(label_space_prob.tolist())
            if run_control_experiment_parallelly:
                ret_control.append(label_space_prob_control.tolist())
    if run_control_experiment_parallelly:
        return ret_experiment, ret_control
    return ret_experiment




## Overloaded functions for the models

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values,
    output_attentions: bool,
    head_mask: torch.Tensor,
    alibi: torch.Tensor,
):
    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
    # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
    # to infer the attention mask.
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    using_static_cache = isinstance(past_key_values, StaticCache)
    # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
    if (
        self.config._attn_implementation == "sdpa"
        and not using_static_cache
        and not output_attentions
        and head_mask is None
        and alibi is None
    ):
        if AttentionMaskConverter._ignore_causal_mask_sdpa(
            attention_mask,
            inputs_embeds=input_tensor,
            past_key_values_length=past_seen_tokens,
            is_training=self.training,
        ):
            return None
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    batch_size, sequence_length, _ = input_tensor.shape
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length
        )
    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
        cache_position=cache_position,
        batch_size=input_tensor.shape[0],
    )
    # We take care to integrate alibi bias in the causal_mask here
    if head_mask is None and alibi is not None:
        alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
        causal_mask = torch.masked_fill(
            alibi / math.sqrt(self.config.hidden_size // self.num_heads),
            causal_mask < -1,
            min_dtype,
        )
    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    return causal_mask


def layered_mask_inference_for_llama_CLM(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
    Returns:
    Example:
    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM
    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")
    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model.layered_mask_inference_for_llama(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()
    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def layered_mask_inference_for_falcon_CLM(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    transformer_outputs = self.transformer.layered_mask_inference_for_falcon(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    hidden_states = transformer_outputs[0]
    lm_logits = self.lm_head(hidden_states)
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
        )
    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output
    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def layered_mask_inference_for_falcon(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
    # Compute alibi tensor: check build_alibi_tensor documentation
    use_legacy_cache = False
    alibi = None
    past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
    batch_size, seq_length, _ = inputs_embeds.shape
    if cache_position is None:
        cache_position = torch.arange(
            past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    causal_mask = []
    for layer in range(self.config.num_hidden_layers):
        causal_mask.append(
            self._update_causal_mask(
                attention_mask[layer].to(self.device), inputs_embeds, cache_position, past_key_values, output_attentions, head_mask, alibi
            )
        )
    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    hidden_states = inputs_embeds
    # create position embeddings to be shared across the decoder layers
    # position_embeddings = self.rotary_emb(hidden_states, position_ids)
    next_decoder_cache = None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, block in enumerate(self.h):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if self.gradient_checkpointing and self.training:
            outputs = self._gradient_checkpointing_func(
                block.__call__,
                hidden_states,
                alibi,
                causal_mask,
                position_ids,
                head_mask[i],
                past_key_values,
                use_cache,
                output_attentions,
                # cache_position,
                # position_embeddings,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=past_key_values,
                attention_mask=causal_mask[i],
                position_ids=position_ids,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
                # cache_position=cache_position,
                # position_embeddings=position_embeddings,
            )
        hidden_states = outputs[0]
        if use_cache is True:
            next_decoder_cache = outputs[1]
        if output_attentions:
            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    if not return_dict:
        return tuple(
            v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def layered_mask_inference_for_llama(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )
    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    return_legacy_cache = False
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    causal_mask = []
    for layer in range(self.config.num_hidden_layers):
        causal_mask.append(
            self._update_causal_mask(
                attention_mask[layer].to(self.device), inputs_embeds, cache_position, past_key_values, output_attentions
            )
        )
    hidden_states = inputs_embeds
    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    for index, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask[index],
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask[index],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    hidden_states = self.norm(hidden_states)
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def layered_mask_inference_for_llamaQ(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )
    if self.gradient_checkpointing and self.training and use_cache:
        use_cache = False
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    return_legacy_cache = False
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    causal_mask = []
    for layer in range(self.config.num_hidden_layers):
        causal_mask.append(
            attention_mask[layer].to(self.device)
        )
    hidden_states = inputs_embeds
    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    for index, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask[index],
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask[index],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    hidden_states = self.norm(hidden_states)
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

