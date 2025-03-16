import os
import re
import json
import jieba
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-72B-Instruct-AWQ")

sampling_params = SamplingParams(temperature=0.1, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

llm = LLM(model="Qwen2.5-72B-Instruct-AWQ", gpu_memory_utilization=0.8, tensor_parallel_size=2, max_model_length=28496, quantization="awq")


def load_dictionary(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

dictionary = load_dictionary("translation_dict.json")

embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

def score_dictionary_entry(entry, prompt_tokens, prompt_embedding):
    mandarin_text = entry["mandarin"]
    entry_embedding = embedding_model.encode(mandarin_text, convert_to_tensor=True)
    similarity = torch.cosine_similarity(prompt_embedding, entry_embedding).item()
    token_overlap = len(set(prompt_tokens) & set(jieba.cut(mandarin_text)))
    return similarity + token_overlap

def retrieve_relevant_entries(prompt, dictionary, top_k=5):
    prompt_tokens = list(jieba.cut(prompt))
    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True)

    scored_entries = []
    for entry in dictionary:
        score = score_dictionary_entry(entry, prompt_tokens, prompt_embedding)
        scored_entries.append((entry, score))

    scored_entries.sort(key=lambda x: x[1], reverse=True)
    return [e[0] for e in scored_entries[:top_k]]

def build_prompt_with_dictionary(prompt, retrieved_entries):
    dict_reference = "\n".join([f"- {entry['mandarin']} → {entry['cantonese'][0]}" for entry in retrieved_entries])

    enhanced_system_prompt = f"""
    ## Role: Cantonese Translation Expert
    ## Profile:
    - author: Machine Translation   
    - version: 0.1
    - language: Cantonese
    - description: I am an expert in translating Mandarin to Cantonese.
    ## Goals:
    - Translate the input Mandarin text into Cantonese expressions, using Cantonese idiomatic expressions and vocabulary. Only translate, do not provide explanations.
    ## Constraints:
    - No additional explanations or suggestions.
    - Do not unnecessarily convert simplified Chinese characters to traditional Chinese.
    - Do not answer questions within the input text.
    - No English should appear in the output.
    - Do not repeat the input text.
    - Ignore grammatical errors in the input Mandarin text.
    - Output format: <Cantonese>Cantonese content</Cantonese>
    ## Skills:
    - Expert in Cantonese translation
    ## Examples:
    [input]: 你饿不饿，我煮碗面给你吃啊？
    [answer]: <\Cantonese>你饿唔饿啊，我煮碗面俾你食啊？<\Cantonese>

    [input]: 乌鲁木齐的酒店有什么？
    [answer]: <\Cantonese>乌鲁木齐嘅酒店有啲咩？<\Cantonese>
    ## Dictionary Reference:
    You can refer to the following dictionary entries for translation:
    {dict_reference}

    Please translate the following Mandarin text into Cantonese, referring to the dictionary entries where possible. Output only the Cantonese translation without additional explanations.
    """
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages

def translate_with_rag(prompt, dictionary):
    retrieved_entries = retrieve_relevant_entries(prompt, dictionary)

    messages = build_prompt_with_dictionary(prompt, retrieved_entries)

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text], sampling_params)
    translation = output[0].outputs[0].text

    match = re.search(r"<Cantonese>(.*?)</Cantonese>", translation)
    cantonese_text = match.group(1).strip() if match else translation.strip()

    return cantonese_text

def translate_data(input_file, output_file, dictionary, resume=False):
    df = pd.read_csv(input_file, sep='\t')

    translated_files = set()
        
    with open(output_file, 'a', encoding='utf-8', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        if not resume or os.path.getsize(output_file) == 0:
            writer.writerow(["file_name", "prompt", "translation"]) 

        for i, row in df.iterrows():
            file_name, prompt = row[0], row[2]

            if file_name in translated_files:
                continue  

            try:
                translation = translate_with_rag(prompt, dictionary)
                writer.writerow([file_name, prompt, translation])
                logging.info(f"[{i+1}] 完成翻译: {file_name} --> {translation}")
            except Exception as e:
                logging.error(f"[{i+1}] 翻译失败: {file_name}, 错误: {e}")
                writer.writerow([file_name, prompt, "翻译失败"])

if __name__ == "__main__":
    input_file = "translated_yuetrain.tsv"
    output_file = "trans_yuetrain3.tsv"
    translate_data(input_file, output_file, dictionary, resume=True)