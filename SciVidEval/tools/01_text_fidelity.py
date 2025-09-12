import os
import json
import torch
import re
import http.client
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from rouge_score import rouge_scorer
from tqdm import tqdm
from collections import defaultdict

# --- 1. 配置区域 (Configuration Area) ---

# 定义所有基线模型/数据源的信息
# - key: 基线名称，将用于输出文件名
# - path: 数据源的根目录
# - file_pattern: 文件名的格式化模板，{paper_name} 会被替换
# - format: JSON文件的读取格式 ('paper', 'timed_script', 'direct_values')
BASELINES = {
    "Paper": {
        "path": "/data/ssd/Data/VideoAgent/paper_content/",
        "file_pattern": "{paper_name}_raw_content_v3.json",
        "format": "paper"
    },
    "Human": {
        "path": "/data/ssd/Data/Oral/processed/",
        "file_pattern": "{paper_name}/{paper_name}_gpt.json",
        "format": "timed_script"
    },
    "Lunwenshuo": {
        "path": "/data/ssd/Data/LunWenShuo/processed/",
        "file_pattern": "{paper_name}/{paper_name}_gpt.json",
        "format": "timed_script"
    },
    "Pictory": {
        "path": "/data/ssd/Data/Pictory/processed/",
        "file_pattern": "{paper_name}/{paper_name}_gpt.json",
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

# 论文原文的基线名称 (用作评估的Ground Truth)
PAPER_REFERENCE_KEY = "Paper"

# PPL 计算配置
PPL_MODEL_NAME = './gpt2-large'  # 推荐使用本地路径，避免重复下载

# LLM API 配置
API_HOST = ""
API_KEY = "Bearer sk-"  # !! 请替换成您自己的API Key !!
LLM_MODEL_NAME = "gpt-4-turbo"

# 输出结果的目录
RESULTS_DIR = "./results"

# --- 2. 全局模型加载 (Global Model Loading) ---

print("Initializing evaluation models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_NAME).to(device)
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_NAME)
    print(f"Perplexity model '{PPL_MODEL_NAME}' loaded successfully.")
except OSError:
    print(f"Error: Could not find PPL model at '{PPL_MODEL_NAME}'.")
    print("Please ensure the model is downloaded and the path is correct.")
    exit()


# --- 3. 辅助函数 (Helper Functions) ---

def load_and_prepare_text(file_path, file_format):
    """根据指定的格式加载JSON文件并返回拼接好的字符串"""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if file_format == 'paper':
                # 提取论文原文内容
                return " ".join(section['content'] for section in data.get('sections', []) if section.get('content'))
            elif file_format == 'timed_script' or file_format == 'direct_values':
                # 适用于key是时间戳或其它，我们只需要value的场景
                return " ".join(str(v) for v in data.values())
            else:
                print(f"Warning: Unknown file format '{file_format}' for {file_path}")
                return None
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error reading or parsing file {file_path}. Error: {e}")
        return None


def get_paper_names(reference_path, reference_suffix):
    """从参考目录获取所有论文的基础名称列表"""
    paper_names = []
    if not os.path.exists(reference_path):
        print(f"Error: Reference directory not found at '{reference_path}'")
        return []
    for f in os.listdir(reference_path):
        if f.endswith(reference_suffix):
            paper_names.append(f.replace(reference_suffix, ''))
    print(f"Found {len(paper_names)} papers to evaluate.")
    return sorted(paper_names)


# --- 4. 指标计算函数 (Metric Calculation Functions) ---

def calculate_perplexity(text, model, tokenizer):
    """计算给定文本的困惑度 (Perplexity)"""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return float('inf')

    encodings = tokenizer(text, return_tensors='pt')

    # --- FIX ---
    # 使用 tokenizer.model_max_length 作为窗口大小，这是更稳健的做法
    # 避免使用 model.config.max_position_embeddings，因为它可能不总是与 tokenizer 匹配
    max_length = tokenizer.model_max_length
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)

        # 确保切片的长度不会超过模型的最大长度
        if input_ids.size(1) == 0:
            continue

        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # 确保loss不是nan
            if not torch.isnan(outputs.loss):
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    if not nlls:
        return float('inf')

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def calculate_rouge_l(generated_text, ground_truth_text):
    """计算ROUGE-L F1分数"""
    if not generated_text or not ground_truth_text:
        return 0.0
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth_text, generated_text)
    return scores['rougeL'].fmeasure


def get_llm_consistency_score(generated_narration, source_paper_text):
    """使用LLM评估解说脚本与论文原文的语义一致性"""
    if not generated_narration or not source_paper_text:
        return -1.0

    # 为避免prompt过长，可以对原文和生成文本进行截断
    source_paper_text = source_paper_text[:8000]
    generated_narration = generated_narration[:4000]

    prompt = f"""
You are an expert academic reviewer evaluating how well an AI-generated narration script summarizes a scientific paper.

**Task:** Evaluate the semantic consistency between the "Generated Narration" and the "Source Paper Content".
- Does the narration accurately reflect the paper's core concepts, methodology, and key findings?
- Does it maintain factual accuracy and logical flow according to the paper?
- Ignore minor differences in wording or simplification for a general audience, as long as the core meaning is preserved.

**Source Paper Content (Excerpt):**
"{source_paper_text}"

**Generated Narration:**
"{generated_narration}"

Provide a consistency score from 1 to 10, where:
1 = Completely inconsistent, inaccurate, or contradictory.
5 = Partially consistent, but misses key points or contains inaccuracies.
10 = Perfectly consistent, accurately and faithfully conveying the paper's essential information.

**IMPORTANT: Respond with ONLY a single number and nothing else.**
"""

    conn = http.client.HTTPSConnection(API_HOST)
    payload = json.dumps({
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 10
    })
    headers = {'Accept': 'application/json', 'Authorization': API_KEY, 'Content-Type': 'application/json'}

    try:
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_json = json.loads(data.decode("utf-8"))
        content = response_json['choices'][0]['message']['content']
        match = re.search(r'\d+\.?\d*', content)
        return float(match.group()) if match else -1.0
    except Exception as e:
        print(f"  - LLM API Error: {e}")
        return -1.0
    finally:
        conn.close()


# --- 5. 主处理逻辑 (Main Processing Logic) ---

def main():
    """主函数，执行所有评估流程"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    paper_ref_config = BASELINES[PAPER_REFERENCE_KEY]
    paper_names = get_paper_names(
        paper_ref_config['path'],
        paper_ref_config['file_pattern'].split('{paper_name}')[-1]
    )

    if not paper_names:
        print("No papers found to evaluate. Exiting.")
        return

    for baseline_name, config in BASELINES.items():
        print(f"\n===== Processing Baseline: {baseline_name} =====")

        baseline_results = {}
        # 用于计算平均分的列表
        scores = defaultdict(list)

        for paper_name in tqdm(paper_names, desc=f"Evaluating {baseline_name}"):
            # 1. 加载论文原文
            paper_file_path = os.path.join(
                paper_ref_config['path'],
                paper_ref_config['file_pattern'].format(paper_name=paper_name)
            )

            paper_text = load_and_prepare_text(paper_file_path, paper_ref_config['format'])
            if not paper_text:
                print(f"Warning: Could not load reference paper for {paper_name}. Skipping.")
                continue

            # 2. 加载当前基线的文本
            baseline_file_path = os.path.join(
                config['path'],
                config['file_pattern'].format(paper_name=paper_name)
            )
            print(baseline_file_path)
            baseline_text = load_and_prepare_text(baseline_file_path, config['format'])

            if not baseline_text:
                print(f"Warning: Could not find {baseline_name} for {paper_name}, Skipping")
                # 文件缺失，直接跳过这篇论文的计算
                continue

            # 3. 计算所有指标
            if baseline_name == 'Paper': continue
            ppl_score = calculate_perplexity(baseline_text, ppl_model, ppl_tokenizer)
            rouge_l_score = calculate_rouge_l(baseline_text, paper_text)
            llm_judge_score = get_llm_consistency_score(baseline_text, paper_text)

            # 4. 存储单篇论文的结果
            paper_result = {
                'PPL': ppl_score,
                'Rouge-L': rouge_l_score,
                'LLM-Judge': llm_judge_score
            }
            baseline_results[paper_name] = paper_result

            # 将有效分数添加到平均分列表
            if ppl_score != float('inf'): scores['PPL'].append(ppl_score)
            if rouge_l_score >= 0.0: scores['Rouge-L'].append(rouge_l_score)  # Rouge can be 0
            if llm_judge_score >= 0.0: scores['LLM-Judge'].append(llm_judge_score)  # LLM score can be 0

        # 5. 计算并存储平均分
        if scores:
            average_scores = {
                metric: np.mean(values) if values else 0.0
                for metric, values in scores.items()
            }
            baseline_results['average_scores'] = average_scores
            print(f"--- Average Scores for {baseline_name} ---")
            print(json.dumps(average_scores, indent=2))
        else:
            print(f"No valid files found for baseline: {baseline_name}")

        # 6. 将该基线的所有结果写入JSON文件
        output_filename = os.path.join(RESULTS_DIR, f'narration_quality_{baseline_name}.json')
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(baseline_results, f, indent=4)
        print(f"✅ Results for {baseline_name} saved to '{output_filename}'")


if __name__ == '__main__':
    main()

