from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch
import pandas as pd
import json
from tqdm import tqdm
import outlines


#### First clause 

def gen_first_clause(model, tokenizer):

    with open("./prompt/first_clause_gen.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # load idioms
    df = pd.read_csv("../data/springer_idiom.csv")
    idioms = df["idiom"].tolist()

    json_schema = '''{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Outlines",
    "type": "object",
    "properties": {
        "idiom": {"title": "Idiom", "type": "string"},
        "first_clause": {"title": "First Clause", "type": "string"},
        "figurative_word": {"title": "Figurative Token", "type": "string"},
        "literal_word": {"title": "Literal Token", "type": "string"}
    },
    "required": ["idiom", "first_clause", "figurative_word", "literal_word"]
    }'''


    first_clause_data = []
    for idiom in tqdm(idioms):
        prompt = f'Instruct: {text}\n\nInput: {idiom}\nOutput:'
        message_history = [{"role": "system", "content": "You are an expert of idiom. Provide concise answers, focusing on the key information needed."},{"role": "user", "content": prompt}]
        message = tokenizer.apply_chat_template(message_history, add_generation_prompt=True)
        generator = outlines.generate.json(model, json_schema)
        instance = generator(tokenizer.decode(message))

        first_clause_data.append(instance)


    with open('../data/first_clause.json', "w", encoding="utf-8") as file:
        json.dump(first_clause_data, file, ensure_ascii=False, indent=4)



#### Paraphrase
        
def gen_paraphrase(model, tokenizer):

    with open('../data/first_clause.json', "r", encoding="utf-8") as file:
        first_clause_data = json.load(file)
    
    with open("./prompt/paraphrase.txt", "r", encoding="utf-8") as file:
        text = file.read()
    
    paraphrased_data = []
    for data in tqdm(first_clause_data):
        idiom = data["idiom"]
        original_sentence = data["first_clause"]

        prompt = f'Instruct: {text}\n\nInput: idiom: {idiom} original_sentence: {original_sentence}\nOutput:'

        json_schema = '''{
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Outlines",
            "type": "object",
            "properties": {
                "idiom": {"title": "Idiom", "type": "string"},
                "original_sentence": {"title": "Original Sentence", "type": "string"},
                "figurative_sentence": {"title": "Figurative Sentence", "type": "string"},
                "literal_sentence": {"title": "Literal Sentence", "type": "string"}
            },
            "required": ["idiom", "original_sentence", "figurative_sentence", "literal_sentence"]
            }'''
        
        message_history = [{"role": "system", "content": "You are an expert of idiom. Provide concise answers, focusing on the key information needed."},{"role": "user", "content": prompt}]
        message = tokenizer.apply_chat_template(message_history, add_generation_prompt=True)
        generator = outlines.generate.json(model, json_schema)

        output = generator(tokenizer.decode(message))
        paraphrased_data.append(output)

    with open('../data/paraphrased.json', "w", encoding="utf-8") as file:
        json.dump(paraphrased_data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":

# load model
    # model_id = "Qwen/QwQ-32B"
    model_id = 'meta-llama/Llama-3.3-70B-Instruct'
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = outlines.models.transformers(model_id, model_kwargs={'device_map': "auto", 'torch_dtype': torch.bfloat16, 'quantization_config' : quantization_config})
    # model = outlines.models.transformers(model_id, model_kwargs={'device_map': "auto", 'torch_dtype': torch.bfloat16})
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # gen_first_clause(model, tokenizer)
    # print('Done with first clause!')
    gen_paraphrase(model, tokenizer)
    print('Done with paraphrase!')




