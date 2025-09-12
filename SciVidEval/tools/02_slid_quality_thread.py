import os
import glob
import json
import torch
import re
import http.client
import base64
import concurrent.futures
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from natsort import natsorted  # For natural sorting of filenames like 'frame_5.jpg', 'frame_10.jpg'

# --- 1. 全局配置 (请根据您的环境修改) ---

# API 和模型配置
API_HOST = "
API_KEY = "Bearer "  # <--- 在这里填入你的API Key
LLM_MODEL_NAME = "gpt-4o"
PPL_MODEL_NAME = './gpt2-large'  # 使用本地的gpt2-large模型

# 并发配置
# 设置同时发送的API请求数量。可根据您的网络和API限制在 10 到 50 之间调整。
MAX_WORKERS = 20

# 基础数据目录
BASE_DIR = "/data/ssd/Data"
PAPER_DIR = os.path.join(BASE_DIR, "VideoAgent/paper_content")

# 定义所有需要遍历的 baseline 及其路径规则
BASELINES = {
    "Human": {
        "path": os.path.join(BASE_DIR, "Oral/processed/{poster_name}/frame"),
        "file_pattern": "frame_*.jpg"
    },
    "Pictory": {
        "path": os.path.join(BASE_DIR, "Pictory/processed/{poster_name}/frame"),
        "file_pattern": "frame_*.jpg"
    },
    "LunWenshuo": {
        "path": os.path.join(BASE_DIR, "LunWenShuo/processed/{poster_name}/frame"),
        "file_pattern": "frame_*.jpg"
    },
    "4o": {
        "path": os.path.join(BASE_DIR, "VideoAgent/generated_frame_4o/{poster_name}_multipage"),
        "file_pattern": "*.jpg"
    },
    "Gemini": {
        "path": os.path.join(BASE_DIR, "VideoAgent/generated_frame_gemini/{poster_name}_multipage"),
        "file_pattern": "*.jpg"
    },
    "Qwen": {
        "path": os.path.join(BASE_DIR, "VideoAgent/generated_frame_qwen-2.5-vl-7b/{poster_name}_multipage"),
        "file_pattern": "*.jpg"
    }
}

# --- 2. 全局模型加载 (避免在循环中重复加载) ---

print(f"Loading Perplexity model: {PPL_MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_NAME).to(device)
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_NAME)
    print("PPL model loaded successfully.")
except Exception as e:
    print(f"Error loading PPL model: {e}")
    ppl_model, ppl_tokenizer = None, None


# --- 3. 核心 API 调用与计算函数 ---

def get_image_base64(image_path):
    """将图片文件转换为 base64 编码的字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Error reading image file {image_path}: {e}")
        return None


def call_llm_api(prompt, image_path=None):
    """
    调用多模态 LLM API 的通用函数。
    如果提供了 image_path，则执行多模态调用。
    """
    content = [{"type": "text", "text": prompt}]
    if image_path:
        base64_image = get_image_base64(image_path)
        if base64_image:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        else:
            return None  # 图像读取失败

    payload = json.dumps({
        "model": LLM_MODEL_NAME,
        "stream": False,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2000,
        "temperature": 0.1  # 使用较低的温度以获得更稳定的、可复现的结果
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': API_KEY,
        'Content-Type': 'application/json'
    }

    try:
        conn = http.client.HTTPSConnection(API_HOST, timeout=60)
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()

        if res.status != 200:
            print(f"API request failed with status {res.status}: {res.read().decode('utf-8')}")
            return None

        data = res.read()
        response_json = json.loads(data.decode("utf-8"))

        if 'choices' in response_json and len(response_json['choices']) > 0:
            return response_json['choices'][0]['message']['content']
        else:
            print(f"API response missing 'choices': {response_json}")
            return None
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()


def extract_text_from_image(image_path):
    """功能 1: 提取图片中的文本内容"""
    prompt = "You are an OCR (Optical Character Recognition) tool. Extract all text from the provided image. Respond only with the extracted text, without any additional comments or explanations."
    return call_llm_api(prompt, image_path)


def evaluate_image_text_match(image_path, page_text, paper_text):
    """功能 2: 评估PPT中图像和文本内容是否匹配"""
    if not image_path:
        return None
    prompt = f"""
As an academic reviewer, evaluate if the slide's image visually corresponds to the slide's text content, in the context of the full research paper.
- If the text describes a 'network architecture', the image should show a diagram of it.
- If the text presents 'quantitative results', a relevant table or chart is expected.
- If the image is purely decorative or a simple title page, it's a non-match unless the text is also just a title.

Respond with a single number: 1 for a good match, 0 for a poor match. Do not provide any other text.

**Full Paper Content:**
---
{paper_text[:8000]}
---
**Slide Text:**
---
{page_text}
---
"""
    response = call_llm_api(prompt, image_path)
    if response:
        match = re.search(r'\b[01]\b', response)
        if match:
            return int(match.group(0))
    return 0  # 默认不匹配


def evaluate_consistency(page_text, paper_text):
    """功能 3: 评估PPT和论文原文对应章节的一致性"""
    prompt = f"""
As an academic reviewer, evaluate the semantic consistency between a presentation slide's text and the full research paper.
Does the slide text accurately summarize or represent information present in the paper?
Provide a score from 0 (completely inconsistent) to 10 (perfectly consistent and accurate summary).
Respond ONLY with a single number.

**Full Paper Content:**
---
{paper_text[:8000]}
---
**Slide Text:**
---
{page_text}
---
"""
    response = call_llm_api(prompt)
    if response:
        match = re.search(r'\b(\d(\.\d+)?|10)\b', response)
        if match:
            return float(match.group(0))
    return 0.0  # 默认0分


def calculate_perplexity(text, model, tokenizer):
    """功能 4: 计算给定文本的困惑度 (Perplexity)"""
    if not text or not model or not tokenizer:
        return float('inf')

    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            if outputs.loss is not None:
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if not nlls:
        return float('inf')

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


# --- 4. 主处理逻辑 ---

def process_single_image(args):
    """
    (供并发调用) Worker function: 处理单张图片的所有评估任务。
    'args' 是一个元组 (image_path, paper_text)，以便与 executor.map 兼容。
    """
    image_path, paper_text = args
    # 1. 提取文本
    text = extract_text_from_image(image_path)
    if not text:
        text = ""  # 如果提取失败，视为空文本
    # 2. 评估图文匹配度
    match_label = evaluate_image_text_match(image_path, text, paper_text)
    # 3. 评估与原文一致性
    consistency_score = evaluate_consistency(text, paper_text)
    return text, match_label, consistency_score


def process_poster(poster_name, baseline_config, paper_text):
    """处理单个 poster 的所有幻灯片（并发版本）"""
    ppt_dir = baseline_config["path"].format(poster_name=poster_name)
    file_pattern = baseline_config["file_pattern"]
    output_path = os.path.join(ppt_dir, "evaluation_results.json")

    # 如果结果已存在，则跳过，这是您要求的功能
    if os.path.exists(output_path):
        return

    if not os.path.isdir(ppt_dir):
        return

    # 查找、排序并跳过第一张图片
    image_files = glob.glob(os.path.join(ppt_dir, file_pattern))
    sorted_images = natsorted(image_files)

    if len(sorted_images) <= 1:
        return

    images_to_process = sorted_images[1:]

    # --- 并发处理开始 ---
    page_texts, match_labels, consistency_scores = [], [], []
    tasks = [(image_path, paper_text) for image_path in images_to_process]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 executor.map 并发执行任务，并用 tqdm 显示进度条
        results = list(
            tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc=f"Processing {poster_name}",
                 leave=False))

    if results:
        # 解包结果
        page_texts, match_labels, consistency_scores = zip(*results)

    # --- 并发处理结束 ---

    # 4. 计算 PPL (在所有文本获取后进行)
    full_ppt_text = " ".join(page_texts)
    ppl_score = calculate_perplexity(full_ppt_text, ppl_model, ppl_tokenizer)

    # 5. 整合结果并保存
    result_data = {
        "page_texts": list(page_texts),
        "match_labels": list(match_labels),
        "consistency_scores": list(consistency_scores),
        "ppl": ppl_score
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        print(f"Error writing json file to {output_path}: {e}")


# --- 5. 最终统计分析 ---

def calculate_final_statistics():
    """遍历所有json文件，统计最终结果"""
    print("\n" + "=" * 50)
    print("Calculating Final Statistics...")
    print("=" * 50)

    final_results = {}

    for baseline_name, config in BASELINES.items():
        search_path = config["path"].format(poster_name='*')
        json_files = glob.glob(os.path.join(search_path, "evaluation_results.json"))

        if not json_files:
            continue

        total_ppl, total_match_accuracy, total_consistency = 0, 0, 0
        valid_posters = 0

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 检查数据有效性
                if "ppl" not in data or "match_labels" not in data or "consistency_scores" not in data:
                    continue

                total_ppl += data["ppl"]

                if data["match_labels"]:
                    total_match_accuracy += sum(data["match_labels"]) / len(data["match_labels"])
                if data["consistency_scores"]:
                    total_consistency += sum(data["consistency_scores"]) / len(data["consistency_scores"])

                valid_posters += 1

            except (json.JSONDecodeError, IOError, ZeroDivisionError) as e:
                print(f"Error processing {json_file}: {e}")

        if valid_posters > 0:
            final_results[baseline_name] = {
                "average_ppl": total_ppl / valid_posters,
                "average_match_accuracy": total_match_accuracy / valid_posters,
                "average_consistency_score": total_consistency / valid_posters,
                "processed_posters": valid_posters
            }

    # 打印结果
    print("\n--- Final Evaluation Results ---")
    if not final_results:
        print("No results to display. Please check if any JSON files were generated.")
    for baseline, metrics in final_results.items():
        print(f"\nBaseline: {baseline}")
        print(f"  - Processed Posters: {metrics['processed_posters']}")
        print(f"  - Average PPL (lower is better): {metrics['average_ppl']:.4f}")
        print(f"  - Average Match Accuracy (0-1, higher is better): {metrics['average_match_accuracy']:.4f}")
        print(f"  - Average Consistency Score (0-10, higher is better): {metrics['average_consistency_score']:.4f}")

    with open("final_baseline_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4)
    print("\n✅ Final statistics saved to 'final_baseline_statistics.json'")


# --- 6. 主执行入口 ---

if __name__ == '__main__':
    all_posters = [os.path.basename(f).replace('_raw_content_v3.json', '') for f in
                   glob.glob(os.path.join(PAPER_DIR, '*.json'))]

    if not all_posters:
        print(f"Error: No paper files found in '{PAPER_DIR}'. Cannot determine poster names.")
    else:
        print(f"Found {len(all_posters)} unique posters to process.")

        for baseline_name, config in BASELINES.items():
            print(f"\nProcessing Baseline: {baseline_name}...")
            for poster_name in tqdm(all_posters, desc=f"Overall {baseline_name} Progress"):
                paper_file_path = os.path.join(PAPER_DIR, f"{poster_name}_raw_content_v3.json")

                if not os.path.exists(paper_file_path):
                    continue

                try:
                    with open(paper_file_path, 'r', encoding='utf-8') as f:
                        paper_data = json.load(f)
                        paper_text = " ".join([sec['content'] for sec in paper_data.get('sections', [])])
                except (json.JSONDecodeError, IOError) as e:
                    print(f"\nError reading paper file for {poster_name}: {e}. Skipping.")
                    continue

                if not paper_text:
                    continue

                process_poster(poster_name, config, paper_text)

        calculate_final_statistics()
