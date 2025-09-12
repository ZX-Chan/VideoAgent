import base64
import os
import json
from openai import OpenAI
from pathlib import Path
import concurrent.futures
import time
from tqdm import tqdm


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ocr_image(args):
    image_path, api_key, base_url = args
    try:
        base64_image = encode_image(image_path)
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "你是OCR助手。请提取图片中的代码和文本内容。以 markdown的格式,但只返回提取的内容，不要添加任何markdown标记、注释或封装格式。保持原有的换行和缩进。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "提取图片中的文本,直接返回文本内容,不要添加任何格式"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0.7,
            stream=False
        )

        return str(image_path), response.choices[0].message.content
    except Exception as e:
        print(f"处理 {image_path} 时出错: {str(e)}")
        return str(image_path), None


def main():
    API_KEY = "sk"
    BASE_URL = ""
    MAX_WORKERS = 40

    folder_path = Path(".")
    with open(folder_path / "1_4_19_timestamps.json", 'r') as f:
        timestamps_dict = {item['path']: item['timestamp'] for item in json.load(f)['timestamps']}

    image_files = list(folder_path.glob("frame_*.jpg"))
    total_files = len(image_files)
    results = {}
    completed = 0

    print(f"开始处理 {total_files} 个图片文件...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        args = [(str(path), API_KEY, BASE_URL) for path in image_files]
        futures = [executor.submit(ocr_image, arg) for arg in args]

        with tqdm(total=total_files) as pbar:
            for future in concurrent.futures.as_completed(futures):
                image_path, ocr_result = future.result()
                if ocr_result:
                    filename = Path(image_path).name
                    results[filename] = {
                        "text": ocr_result,
                        "timestamp": timestamps_dict.get(filename, "")
                    }
                pbar.update(1)

    with open('ocr_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n处理完成! 结果已保存到 ocr_results.json")


# if __name__ == "__main__":
#     main()

