from openai import OpenAI
import json
from collections import OrderedDict

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="***REMOVED***",
    base_url="***REMOVED***"
)




def parse_time_to_seconds(time_str):
    """
    将时间字符串 'HH:MM:SS.mmm' 转换为总秒数。

    :param time_str: str, 时间字符串，如 '00:18:20.000'
    :return: float, 总秒数
    """
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def format_seconds_to_time(seconds):
    """
    将总秒数转换为 'HH:MM:SS.mmm' 格式的时间字符串。

    :param seconds: float, 总秒数
    :return: str, 格式化后的时间字符串
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# 非流式响应
def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    print(completion.choices[0].message.content)

def gpt_35_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        stream=True,
    )
    result = ''
    for chunk in stream:
        try:
            if chunk.choices[0].delta.content is not None:
                result += chunk.choices[0].delta.content
        except:
            pass
    return result


def gpt_4o_api(messages):
    """
    调用 GPT-4o 多模态模型的 API。

    :param messages: 符合 OpenAI API 格式的 messages 列表。
    :return: 来自模型的响应文本，如果出错则返回错误信息。
    """
    try:
        print("正在调用 GPT-4o API...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
            # max_tokens=1024,  # 您可以按需调整
        )

        # 提取并返回响应内容
        content = response.choices[0].message.content
        print("API 调用成功。")
        return content

    except Exception as e:
        error_message = f"调用 GPT-4o API 时发生错误: {e}"
        print(error_message)
        return error_message

def merge_asr_ocr(asr_dict, ocr_dict):
    """
    合并 ASR 和 OCR 字典为一个新的有序字典，确保 ASR 的时间段早于 OCR 的时间点。

    :param asr_dict: dict，键为 'HH:MM:SS.mmm-HH:MM:SS.mmm' 格式的时间段，值为对应的文本
    :param ocr_dict: dict，键为 'HH:MM:SS.mmm' 格式的时间点，值为对应的文本
    :return: OrderedDict，合并并排序后的字典
    """
    entries = []

    asr_text = list(asr_dict.values())[0]


    start_sec = parse_time_to_seconds("00:00:00.000")
    # 处理 OCR 字典
    for time_str, text in ocr_dict.items():
        texts = text.split("\n")[:10]
        fil_texts = ' '.join(texts)
        try:
            end_sec = parse_time_to_seconds(time_str)
            entries.append({
                'type': 'OCR',
                'start': start_sec,
                'end': end_sec,
                'text': fil_texts
            })
            start_sec = end_sec
        except ValueError:
            print(f"时间点格式错误: {time_str}")
            continue

    # 排序：首先按 start 时间排序，如果 start 相同，ASR 排在 OCR 前
    entries_sorted = sorted(
        entries,
        key=lambda x: x['start']
    )

    ocr_json_str = json.dumps(entries_sorted, indent=2)

    prompt_content = f"""
    You will be given two inputs: a full ASR transcript as a single block of text, and an OCR JSON object containing timed text from presentation slides.

    Your task is to precisely align all ASR text with the corresponding slides, leveraging the timestamps provided in the OCR JSON and then correct any grammatical errors or disfluencies in the ASR text. The process must cover the entire ASR text, and the final output must not contain any repetitions or omissions.
    
    **Inputs:**

    1.  **`[FULL_ASR_TEXT]`**: The complete Automatic Speech Recognition transcript from the video.
    2.  **`[OCR_JSON]`**: A JSON list where each object represents a slide with its start time, end time, and recognized text. The format for each object is: `{{'type': 'OCR', 'start': <float>, 'end': <float>, 'text': '<ocr_text>'}}`.

    **Instructions:**

    1.  Iterate through each object in the `[OCR_JSON]` list.
    2.  For each object, use its `'start'` and `'end'` timestamps as a guide to identify and extract the corresponding segment of speech from the `[FULL_ASR_TEXT]`.
    3.  Assign the identified ASR text segment to that slide.
    4.  Ensure that the ASR text is segmented sequentially and completely, so the text from one slide flows directly into the next without any overlap or gaps.
    5.  Format the result as a new JSON list according to the specified output format.

    **Output Format:**

    The output must be a single JSON list of objects. Do not include any text or explanations outside of the JSON. Each object in the list must contain the following keys:
    * `start`: The start timestamp from the corresponding slide in the input `[OCR_JSON]`.
    * `end`: The end timestamp from the corresponding slide in the input `[OCR_JSON]`.
    * `gpt_text`: The full, extracted ASR text that corresponds to this time segment.

    **Here is the data to process:**

    **FULL_ASR_TEXT:**
    ```
    {asr_text}
    ```

    **OCR_JSON:**
    ```json
    {ocr_json_str}
    ```
    """
    
    



    messages = [{'role': 'user',
                 'content': prompt_content}]

    # print(messages)

    result_str = gpt_4o_api(messages)
    try:
        # 清理可能存在于 LLM 响应中的代码块标记
        if result_str.strip().startswith("```json"):
            result_str = result_str.strip()[7:-3]

        final_result = json.loads(result_str)
        print("\n--- Parsed Final Result ---")
        # ensure_ascii=False 以正确显示中文字符
        # print(json.dumps(final_result, indent=2, ensure_ascii=False))
        merged_dict = {}
        for item in final_result:
            start_time = item['start']
            end_time = item['end']
            key = f"{format_seconds_to_time(start_time)}-{format_seconds_to_time(end_time)}"
            merged_dict[key] = item['gpt_text']


    except json.JSONDecodeError:
        print("\nError: The API response was not in a valid JSON format.")
        print("Received:", result_str)
        quit()



    return merged_dict


def merge_asr_images(asr_dict, base64_json_dict):
    """
    合并 ASR 文本和 Base64 编码的图片数据。
    它将一段时间内的语音文本与该时间段结束时出现的关键帧图片关联起来，
    然后调用多模态模型进行处理。

    :param asr_dict: dict, 键为 'HH:MM:SS.mmm-HH:MM:SS.mmm' 格式的时间段，值为语音文本。
    :param base64_json_dict: dict, 键为代表秒数的时间戳(字符串形式)，值为图片的Base64编码字符串。
    :return: dict, 合并并由多模态模型处理后的结果。
    """
    entries = []

    # 1. 处理 ASR 字典，转换为秒和标准格式
    for interval, text in asr_dict.items():
        try:
            start_str, end_str = interval.split('-')
            start_sec = parse_time_to_seconds(start_str)
            end_sec = parse_time_to_seconds(end_str)
            entries.append({
                'type': 'ASR',
                'start': start_sec,
                'end': end_sec,
                'content': text  # 内容是文本
            })
        except ValueError:
            print(f"跳过格式错误的ASR时间段: {interval}")
            continue

    # 2. 处理 Base64 图片字典
    for time_sec_str, base64_data in base64_json_dict.items():
        try:
            time_sec = parse_time_to_seconds(time_sec_str)
            entries.append({
                'type': 'IMAGE',
                'start': time_sec,
                'end': time_sec,
                'content': base64_data  # 内容是Base64字符串
            })
        except (ValueError, TypeError):
            print(f"跳过格式错误的图片时间戳: {time_sec_str}")
            continue

    # 3. 排序：首先按开始时间排序，如果时间相同，ASR优先
    if not entries:
        print("没有有效的数据可供处理。")
        return {}

    entries_sorted = sorted(
        entries,
        key=lambda x: (x['start'], 0 if x['type'] == 'ASR' else 1)
    )

    # 4. 遍历排序后的条目，合并内容并调用API
    merged_results = {}
    accumulated_asr_text = ''
    segment_start_time = entries_sorted[0]['start']

    for entry in entries_sorted:
        if entry['type'] == 'ASR':
            # 如果是语音，累加文本
            accumulated_asr_text += entry['content'] + "\n"

        elif entry['type'] == 'IMAGE':
            # 如果是图片，代表一个内容块的结束
            segment_end_time = entry['end']
            key = f"{format_seconds_to_time(segment_start_time)}-{format_seconds_to_time(segment_end_time)}"
            base64_image = entry['content']

            # 准备发送给多模态模型的数据
            messages = [{'role': 'user',
                        'content': [{
                            "type": "text",
                            "text": "Here is the merged text from a paper presentation video\'s ASR "
                                                                "and PPT. Please extract only the ASR (spoken) content, "
                                                                "precisely and in its entirety. The extracted speech must align with "
                                                                "the corresponding PPT slide, but do not include any text from the slides "
                                                                f"in the output. \nASR content: {accumulated_asr_text}"}]
            }]

            messages[0]['content'].append({
                "image_url": f"data:image/jpeg;base64,{base64_image}"
            })

            # 调用多模态模型API
            gpt_result = gpt_4o_api(messages)

            # 存储结果
            merged_results[key] = gpt_result

            # 重置，为下一个分段做准备
            accumulated_asr_text = ''
            segment_start_time = entry['end']

    return merged_results


if __name__ == '__main__':
    messages = [{'role': 'user', 'content': '鲁迅和周树人的关系'}]

    video_name = 'ICMR2025_01'
    asr_json_path = f'./temp/{video_name}/{video_name}_asr.json'
    ocr_json_path = f'./temp/{video_name}/{video_name}_ocr.json'
    base64_json_path = f'./temp/{video_name}/{video_name}_frames_base64.json'

    # # 读取 OCR JSON 文件
    with open(ocr_json_path, 'r', encoding='utf-8') as ocr_file:
        ocr_data = json.load(ocr_file)

    # 读取 ASR JSON 文件
    with open(asr_json_path, 'r', encoding='utf-8') as asr_file:
        asr_data = json.load(asr_file)

    # 读取 Base64 JSON 文件
    # with open(base64_json_path, 'r', encoding='utf-8') as base64_file:
    #     base64_data = json.load(base64_file)

    result = merge_asr_ocr(asr_data, ocr_data)
    # result = merge_asr_images(asr_data, base64_data)
    print(result)
