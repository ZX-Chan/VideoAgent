import os
import glob
import json
import torch
import re
import http.client
import base64
import concurrent.futures
from tqdm import tqdm
from transformers import AltCLIPModel, AltCLIPProcessor
from PIL import Image
from natsort import natsorted
import torch.nn.functional as F

# --- 1. 全局配置 (请根据您的环境修改) ---

# API 和模型配置
API_HOST = ""
API_KEY = "Bearer sk-"  # <--- 在这里填入你的API Key
LLM_MODEL_NAME = "gpt-4o"
ALTCLIP_MODEL_NAME = 'BAAI/AltCLIP'

# 并发配置
MAX_WORKERS = 20  # 同时发送的API请求/模型处理数量

# 基础数据目录
BASE_DIR = "/data/ssd/Data"

# --- 数据源配置 ---

# 图像数据源
IMAGE_BASELINES = {
    "Human": {
        "path": os.path.join(BASE_DIR, "Oral/processed/{poster_name}/frame"),
        "file_pattern": "frame_*.jpg"
    },
    "Pictory": {
        "path": os.path.join(BASE_DIR, "Pictory/processed/{poster_name}/frame"),
        "file_pattern": "frame_*.jpg"
    },
    "LunWenshuo": {  # 注意：这里名称与TEXT_BASELINES保持一致
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

# 文本数据源
TEXT_BASELINES = {
    "Paper": {
        "path": os.path.join(BASE_DIR, "VideoAgent/paper_content/"),
        "file_pattern": "{poster_name}_raw_content_v3.json",
        "format": "paper"
    },
    "Human": {
        "path": os.path.join(BASE_DIR, "Oral/processed/"),
        "file_pattern": "{poster_name}/{poster_name}_gpt.json",
        "format": "timed_script"
    },
    "LunWenshuo": {  # 注意：这里名称与IMAGE_BASELINES保持一致
        "path": os.path.join(BASE_DIR, "LunWenShuo/processed/"),
        "file_pattern": "{poster_name}/{poster_name}_gpt.json",
        "format": "timed_script"
    },
    "Pictory": {
        "path": os.path.join(BASE_DIR, "Pictory/processed/"),
        "file_pattern": "{poster_name}/{poster_name}_gpt.json",
        "format": "timed_script"
    },
    "GPT-4o": {
        "path": "/data/ssd/Data/VideoAgent/generated_narration_4o/",
        "file_pattern": "{paper_name}_narration.json",
        "format": "direct_values"
    },
    "Gemini": {
        "path": "/data/ssd/Data/VideoAgent/generated_narration_gemini/",
        "file_pattern": "{paper_name}_narration.json",
        "format": "direct_values"
    },
    "Qwen": {
        "path": "/data/ssd/Data/VideoAgent/generated_narration_qwen-2.5-vl-7b/",
        "file_pattern": "{paper_name}_narration.json",
        "format": "direct_values"
    }
}

# --- 2. 全局模型加载 ---

print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    altclip_processor = AltCLIPProcessor.from_pretrained(ALTCLIP_MODEL_NAME)
    altclip_model = AltCLIPModel.from_pretrained(ALTCLIP_MODEL_NAME).to(device)
    print("AltCLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading AltCLIP model: {e}. Please ensure you have an internet connection and the right libraries.")
    altclip_model, altclip_processor = None, None


# --- 3. 核心 API 调用与计算函数 ---

def get_image_base64(image_path):
    """将图片文件转换为 base64 编码的字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"Error reading image file {image_path}: {e}")
        return None


def call_llm_api(prompt, image_path):
    """调用多模态 LLM API 的通用函数"""
    content = [{"type": "text", "text": prompt}]
    base64_image = get_image_base64(image_path)
    if base64_image:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    else:
        return None

    payload = json.dumps({
        "model": LLM_MODEL_NAME, "stream": False,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100, "temperature": 0.1
    })
    headers = {'Accept': 'application/json', 'Authorization': API_KEY, 'Content-Type': 'application/json'}

    try:
        conn = http.client.HTTPSConnection(API_HOST, timeout=60)
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        if res.status != 200:
            print(f"API request failed with status {res.status}: {res.read().decode('utf-8')}")
            return None
        data = res.read()
        response_json = json.loads(data.decode("utf-8"))
        return response_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"An error occurred during API call for {image_path}: {e}")
        return None
    finally:
        if 'conn' in locals(): conn.close()


def calculate_altclip_similarity(image_path, text):
    """功能2: 使用AltCLIP分别提取图片和文本的特征，计算相似度score"""
    if not altclip_model or not altclip_processor: return 0.0
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = altclip_processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = altclip_model(**inputs)

        # 归一化特征并计算余弦相似度
        image_embeds = F.normalize(outputs.image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)

        similarity = F.cosine_similarity(image_embeds, text_embeds, dim=-1).cpu().item()
        return (similarity + 1) / 2  # 将[-1, 1]范围映射到[0, 1]
    except Exception as e:
        print(f"Error during AltCLIP processing for {image_path}: {e}")
        return 0.0


def evaluate_sync_consistency_llm(image_path, text):
    """功能3: 使用API评估图片和文本的一致性分数（0~10）"""
    prompt = f"""
As a multimedia content analyst, your task is to evaluate the synchronization between a visual frame (image) and its corresponding narration script (text).

- Assess if the image visually represents the concepts, objects, or data mentioned in the text.
- Consider the context. For "Figure 3 shows...", the image must be Figure 3. For "We propose a new model...", a diagram is highly relevant.
- A decorative image with substantial text is a mismatch. A title slide with title text is a match.

Provide a consistency score from 0 (completely irrelevant) to 10 (perfectly synchronized and illustrative).
Respond ONLY with a single number.

**Narration Text for this Frame:**
---
{text}
---
"""
    response = call_llm_api(prompt, image_path)
    if response:
        match = re.search(r'\b(\d(\.\d+)?|10)\b', response)
        if match:
            return float(match.group(0))
    return 0.0


# --- 4. 辅助函数与主处理逻辑 ---

def load_text_script(file_path, format_type):
    """根据指定的格式加载和解析文本JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if format_type == "paper":
            return [sec.get('content', '') for sec in data.get('sections', [])]
        elif format_type == "timed_script":
            return [item for key, item in data.items()]
        elif format_type == "direct_values":
            return list(data.values())
        else:
            return []
    except (IOError, json.JSONDecodeError) as e:
        print(f"Could not read or parse text file {file_path}: {e}")
        return []


def process_single_pair(args):
    """(供并发调用) Worker function: 处理单个图文对的评估任务"""
    image_path, text = args
    altclip_score = calculate_altclip_similarity(image_path, text)
    llm_score = evaluate_sync_consistency_llm(image_path, text)
    return altclip_score, llm_score


def process_run(img_baseline_name, txt_baseline_name, poster_name):
    """处理一个完整的 (图像baseline, 文本baseline, poster) 组合"""
    img_config = IMAGE_BASELINES[img_baseline_name]
    txt_config = TEXT_BASELINES[txt_baseline_name]

    img_dir = img_config["path"].format(poster_name=poster_name)
    output_path = os.path.join(img_dir, f"sync_results_{txt_baseline_name}.json")
    # if os.path.exists(output_path): return

    txt_path = os.path.join(txt_config["path"], txt_config["file_pattern"].format(poster_name=poster_name))

    if not os.path.isdir(img_dir) or not os.path.exists(txt_path): return

    images = natsorted(glob.glob(os.path.join(img_dir, img_config["file_pattern"])))
    texts = load_text_script(txt_path, txt_config["format"])

    # 跳过第一张图，并确保文本数量与剩余图片数量一致
    images_to_process = images


    # --- 并发处理 ---
    tasks = list(zip(images_to_process, texts))
    altclip_scores, llm_scores = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(
            tqdm(executor.map(process_single_pair, tasks), total=len(tasks), desc=f"Sync Eval: {poster_name}",
                 leave=False))

    if results:
        altclip_scores, llm_scores = zip(*results)

    # --- 保存结果 ---
    avg_altclip = sum(altclip_scores) / len(altclip_scores) if altclip_scores else 0
    avg_llm = sum(llm_scores) / len(llm_scores) if llm_scores else 0

    result_data = {
        "image_source": img_baseline_name, "text_source": txt_baseline_name, "poster_name": poster_name,
        "average_altclip_similarity": avg_altclip, "average_llm_consistency": avg_llm,
        "details": [{"image": os.path.basename(img), "text": txt, "altclip_score": a_score, "llm_score": l_score}
                    for img, txt, a_score, l_score in zip(images_to_process, texts, altclip_scores, llm_scores)]
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)


# --- 5. 最终统计 ---
def calculate_final_sync_statistics():
    """遍历所有json文件，统计最终结果"""
    print("\n" + "=" * 50 + "\nCalculating Final Synchronization Statistics...\n" + "=" * 50)
    final_results = {}
    for img_baseline_name, img_config in IMAGE_BASELINES.items():
        search_path = img_config["path"].format(poster_name='*')
        json_files = glob.glob(os.path.join(search_path, "sync_results_*.json"))

        for f in json_files:
            try:
                with open(f, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                txt_baseline_name = data['text_source']
                key = f"{img_baseline_name} (img) vs {txt_baseline_name} (txt)"
                if key not in final_results:
                    final_results[key] = {'altclip': [], 'llm': [], 'count': 0}
                final_results[key]['altclip'].append(data['average_altclip_similarity'])
                final_results[key]['llm'].append(data['average_llm_consistency'])
                final_results[key]['count'] += 1
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Error processing result file {f}: {e}")

    print("\n--- Final Synchronization Results ---")
    summary = {}
    for key, values in final_results.items():
        avg_altclip = sum(values['altclip']) / len(values['altclip']) if values['altclip'] else 0
        avg_llm = sum(values['llm']) / len(values['llm']) if values['llm'] else 0
        summary[key] = {"average_altclip_similarity": avg_altclip, "average_llm_consistency": avg_llm,
                        "processed_posters": values['count']}
        print(f"\nCombination: {key}")
        print(f"  - Processed Posters: {values['count']}")
        print(f"  - Avg AltCLIP Similarity (0-1, higher is better): {avg_altclip:.4f}")
        print(f"  - Avg LLM Consistency (0-10, higher is better): {avg_llm:.4f}")

    with open("final_sync_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print("\n✅ Final sync statistics saved to 'final_sync_statistics.json'")


# --- 6. 主执行入口 ---
if __name__ == '__main__':
    all_posters = [os.path.basename(f).replace('_raw_content_v3.json', '') for f in
                   glob.glob(os.path.join(TEXT_BASELINES['Paper']['path'], '*.json'))]
    print(f"Found {len(all_posters)} unique posters to process.")

    # 遍历所有图像源
    for img_b_name in IMAGE_BASELINES.keys():
        # 为每个图像源找到匹配的文本源 (同名)
        if img_b_name in TEXT_BASELINES:
            print(f"\n--- Processing Matched Pair: [{img_b_name}] ---")
            for poster in tqdm(all_posters, desc=f"Pair: {img_b_name}"):
                try:
                    process_run(img_b_name, img_b_name, poster)
                except:
                    continue

        # 额外：让每个图像源都和 'Paper' 文本源进行比较
        # print(f"\n--- Processing Pair: [{img_b_name} (img) vs Paper (txt)] ---")
        # for poster in tqdm(all_posters, desc=f"Pair: {img_b_name} vs Paper"):
        #     process_run(img_b_name, "Paper", poster)

    calculate_final_sync_statistics()
