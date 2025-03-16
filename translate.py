import os
import re
import csv
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-72B-Instruct-AWQ")

sampling_params = SamplingParams(temperature=0.1, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

llm = LLM(model="Qwen2.5-72B-Instruct-AWQ", gpu_memory_utilization=0.8, tensor_parallel_size=2, max_model_length=28496, quantization="awq")

system_content = \
"""
## Role: Translation Expert
## Profile:
- author: Machine Translation
- version: 0.1
- language: Cantonese
- description: I am an expert in translating Mandarin into Cantonese.
## Goals:
- Translate the input Mandarin text into Cantonese expressions, using common Cantonese vocabulary and idioms. Only translate, do not answer.
## Constraints:
- No additional explanations or suggestions.
- Provide alternative translations only if necessary.
- Avoid unnecessary conversion of simplified Chinese characters to traditional Chinese characters.
- Do not answer questions posed in the input text.
- No English should appear in the output.
- Do not repeat the input text.
- Ignore grammatical errors in the input Mandarin text.
- Output format: <Cantonese>Cantonese content</Cantonese>
## Skills:
- Cantonese Expertise
## Examples:
[input]: 你饿不饿，我煮碗面给你吃啊？
[answer]: <Cantonese>你饿唔饿啊，我煮碗面俾你食啊？</Cantonese>
[input]: 乌鲁木齐的酒店有什么？
[answer]: <Cantonese>乌鲁木齐嘅酒店有啲咩？</Cantonese>
"""

def translate_single(prompt):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text], sampling_params)
    translation = output[0].outputs[0].text
    match = re.search(r"<Cantonese>(.*?)</Cantonese>", translation)
    return match.group(1).strip() if match else translation.strip()

def translate_batch(prompts):
    inputs = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs.append(text)

    outputs = llm.generate(inputs, sampling_params)
    results = []
    for output in outputs:
        text = output.outputs[0].text
        match = re.search(r"<Cantonese>(.*?)</Cantonese>", text)
        results.append(match.group(1).strip() if match else text.strip())
    return results

def translate_data(input_file, output_file, resume=False, use_batch=False, batch_size=8, use_threading=False, max_workers=4):
    df = pd.read_csv(input_file, sep='\t')
    translated_files = set()

    if resume and os.path.exists(output_file):
        existing_df = pd.read_csv(output_file, sep='\t')
        translated_files = set(existing_df["file_name"].tolist())
        logging.info(f"已完成 {len(translated_files)} 条记录，将跳过这些数据。")

    with open(output_file, 'a', encoding='utf-8', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        if not resume or os.path.getsize(output_file) == 0:
            writer.writerow(["file_name", "prompt", "translation"])

        rows_to_translate = [
            (row[0], row[2]) for _, row in df.iterrows()
            if row[0] not in translated_files
        ]

        progress_bar = tqdm(total=len(rows_to_translate), desc="翻译进度")

        if use_batch:
            for i in range(0, len(rows_to_translate), batch_size):
                batch = rows_to_translate[i:i+batch_size]
                file_names = [fn for fn, _ in batch]
                prompts = [p for _, p in batch]
                try:
                    translations = translate_batch(prompts)
                    for fn, prompt, trans in zip(file_names, prompts, translations):
                        writer.writerow([fn, prompt, trans])
                except Exception as e:
                    logging.error(f"批量翻译失败: {e}")
                    for fn, prompt in batch:
                        writer.writerow([fn, prompt, "翻译失败"])
                progress_bar.update(len(batch))
        elif use_threading:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {
                    executor.submit(translate_single, prompt): (fn, prompt)
                    for fn, prompt in rows_to_translate
                }
                for future in as_completed(future_to_row):
                    fn, prompt = future_to_row[future]
                    try:
                        trans = future.result()
                    except Exception as e:
                        logging.error(f"翻译失败: {fn}, 错误: {e}")
                        trans = "翻译失败"
                    writer.writerow([fn, prompt, trans])
                    progress_bar.update(1)
        else:
            for fn, prompt in rows_to_translate:
                try:
                    trans = translate_single(prompt)
                except Exception as e:
                    logging.error(f"翻译失败: {fn}, 错误: {e}")
                    trans = "翻译失败"
                writer.writerow([fn, prompt, trans])
                progress_bar.update(1)

        progress_bar.close()

# 启动翻译任务
if __name__ == "__main__":
    input_file = "input.tsv"
    output_file = "output.tsv"

    # 参数控制
    translate_data(
        input_file=input_file,
        output_file=output_file,
        resume=True,
        use_batch=False,
        batch_size=8,
        use_threading=False,
        max_workers=4
    )
