import os
import json
import argparse
import base64
from pathlib import Path
from utils import create_directories, get_video_name, save_timestamps
from utils.asr_extract_audio import extract_audio, audio_to_text
from utils.gpt_rewrite import merge_asr_ocr, merge_asr_images
from utils.frame_cluster import process_frames, get_video_duration, extract_frames
from utils.ocr_extract_ppt import frame2txt



# ==============================================================================
# 核心业务逻辑
# ==============================================================================

def write_log(log_path, message):
    """
    将日志打印到控制台，并追加到日志文件。
    """
    print(message)  # 直接打印到控制台
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"[警告] 写入日志文件失败: {e}")


def init_pipeline(video_path, output_dir, interval):
    """
    初始化整个流程：
    1. 解析输入路径，确定视频名
    2. 创建输出目录结构
    3. 初始化日志文件
    4. 返回包含所有路径和参数的状态字典
    """
    # 1) 检查输入文件是否存在
    input_video_path = Path(video_path)
    if not input_video_path.is_file():
        print(f"[错误] 输入的视频文件路径不存在: {video_path}")
        return None

    # 2) 文件信息
    video_name = get_video_name(video_path)
    print(f"[*] 开始处理视频: {video_name}")

    # 3) 创建输出目录
    output_dir_path = Path(output_dir) / video_name
    output_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[*] 所有输出将保存在: {output_dir_path}")
    audio_dir, frame_dir = create_directories(output_dir_path)

    # 4) 初始化日志
    log_path = output_dir_path / "pipeline_log.txt"
    if log_path.exists():
        log_path.unlink()
    write_log(str(log_path), f"初始化流程: 视频名={video_name}")

    # 5) 构建状态字典，供后续步骤使用
    state = {
        "video_name": video_name,
        "input_video_path": str(input_video_path),
        "output_dir": str(output_dir_path),
        "audio_dir": str(audio_dir),
        "frame_dir": str(frame_dir),
        "interval": interval,
        "log_path": str(log_path),
        "timestamps_json": str(output_dir_path / f"{video_name}_timestamps.json"),
        "base64_json": str(output_dir_path / f"{video_name}_base64.json"),
        "ocr_json": str(output_dir_path / f"{video_name}_ocr.json"),
        "asr_json": str(output_dir_path / f"{video_name}_asr.json"),
        "gpt_json": str(output_dir_path / f"{video_name}_gpt.json")
    }
    with open(str(output_dir_path / f"state.json"), "w") as fp:
        json.dump(state,fp, indent=4)
    return state


def step_extract_audio(state):
    """第一步：提取音频并转录为文本"""
    log_path = state["log_path"]
    write_log(log_path, "\n[步骤 1/5] 开始提取音频并进行ASR转录...")

    output_wav = Path(state["audio_dir"]) / f"{state['video_name']}.wav"
    extract_audio(state["input_video_path"], str(output_wav))

    result_dict, output_asr_path = audio_to_text(str(output_wav))
    state["asr_json"] = output_asr_path  # 更新状态字典中的ASR JSON路径

    msg = f"[成功] 音频提取并转录完成。\n  - 音频文件: {output_wav}\n  - ASR结果: {output_asr_path}"
    write_log(log_path, msg)

    with open(os.path.join(state["output_dir"] , f"state.json"), "w") as fp:
        json.dump(state,fp, indent=4)



def step_extract_frames(state):
    """第二步：从视频中按固定间隔抽帧"""
    log_path = state["log_path"]
    write_log(log_path, f"\n[步骤 2/5] 开始从视频抽帧 (间隔: {state['interval']}秒)...")

    duration = get_video_duration(state["input_video_path"])
    timestamps = extract_frames(
        state["input_video_path"],
        state["frame_dir"],
        state["interval"],
        duration
    )
    save_timestamps(timestamps, state["timestamps_json"])

    msg = f"[成功] 抽帧完成，共提取 {len(timestamps)} 帧。\n  - 帧文件目录: {state['frame_dir']}\n  - 时间戳记录: {state['timestamps_json']}"
    write_log(log_path, msg)



    return timestamps


def step_process_frames(state, timestamps):
    """第三步：对提取的帧进行去重/聚类"""
    log_path = state["log_path"]
    write_log(log_path, "\n[步骤 3/5] 开始对视频帧进行去重/聚类...")

    deduped_frame_paths = process_frames(state)

    # 从路径反查时间戳信息
    deduped_frame_files = {Path(p).name for p in deduped_frame_paths}
    deduped_timestamps = [
        item for item in timestamps if Path(item["path"]).name in deduped_frame_files
    ]

    msg = f"[成功] 帧去重完成，剩余 {len(deduped_timestamps)} 帧。"
    write_log(log_path, msg)
    return deduped_timestamps


def step_ocr_extract(state, deduped_timestamps):
    """第四步：对去重后的帧进行OCR文字提取"""
    log_path = state["log_path"]
    ocr_json_path = state["ocr_json"]
    write_log(log_path, "\n[步骤 4/5] 开始对关键帧进行OCR文字提取...")

    result_dict = {}
    total = len(deduped_timestamps)
    for i, item in enumerate(deduped_timestamps):
        # 简单的进度提示
        if (i + 1) % 20 == 0 or i + 1 == total:
            print(f"  - 正在处理OCR: {i + 1}/{total}")

        frame_file = item["path"]
        t = item["timestamp"]
        text = frame2txt(frame_file, state)
        if text:  # 只保留有文本的结果
            result_dict[t] = text

    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    msg = f"[成功] OCR提取完成。\n  - OCR结果: {ocr_json_path}"
    write_log(log_path, msg)

def image_to_base64(image_path):
    """读取图片文件并将其编码为Base64字符串"""
    try:
        with open(image_path, "rb") as image_file:
            # 读取文件内容，进行base64编码，然后解码成utf-8字符串
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        print(f"  [警告] 图片文件未找到: {image_path}")
        return None


def step_encode_frames(state, deduped_timestamps):
    """第四步：将关键帧图片编码为Base64，用于GPT-4V等多模态模型"""
    log_path = state["log_path"]
    base64_json_path = state["base64_json"]
    write_log(log_path, "\n[步骤 4/5] 开始将关键帧编码为Base64...")

    result_dict = {}
    total = len(deduped_timestamps)
    for i, item in enumerate(deduped_timestamps):
        # 简单的进度提示
        if (i + 1) % 10 == 0 or i + 1 == total:
            print(f"  - 正在编码图片: {i + 1}/{total}")

        frame_file = os.path.join(state['frame_dir'], item['path'])
        print(frame_file)
        t = item["timestamp"]

        base64_string = image_to_base64(frame_file)

        if base64_string:
            result_dict[t] = base64_string

    with open(base64_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f)  # Base64内容较长，不建议用indent=4

    msg = f"[成功] 帧编码完成。\n  - Base64 JSON文件: {base64_json_path}"
    write_log(log_path, msg)


def step_gpt_rewrite_text(state):
    """第五步：结合ASR和OCR结果进行最终整合"""
    log_path = state["log_path"]
    write_log(log_path, "\n[步骤 5/5] 开始合并ASR与OCR结果...")

    try:
        with open(state["ocr_json"], 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        with open(state["asr_json"], 'r', encoding='utf-8') as f:
            asr_data = json.load(f)
    except FileNotFoundError as e:
        write_log(log_path, f"[错误] 无法读取ASR或OCR的JSON文件: {e}")
        return

    result_dict = merge_asr_ocr(asr_data, ocr_data)

    with open(state["gpt_json"], "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    msg = f"[成功] 文本合并完成。\n  - 最终结果: {state['gpt_json']}"
    write_log(log_path, msg)


def step_gpt_rewrite_image(state):
    """第五步：结合ASR和Base64图片数据进行最终整合"""
    log_path = state["log_path"]
    write_log(log_path, "\n[步骤 5/5] 开始合并ASR与图片数据...")

    try:
        with open(state["base64_json"], 'r', encoding='utf-8') as f:
            image_data = json.load(f)
        with open(state["asr_json"], 'r', encoding='utf-8') as f:
            asr_data = json.load(f)
    except FileNotFoundError as e:
        write_log(log_path, f"[错误] 无法读取ASR或Base64的JSON文件: {e}")
        return

    result_dict = merge_asr_images(asr_data, image_data)

    with open(state["gpt_json"], "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    msg = f"[成功] 文本与图片数据合并完成。\n  - 最终结果: {state['gpt_json']}"
    write_log(log_path, msg)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频处理命令行工具')
    parser.add_argument('--input_video', type=str, default='./ICMR2025_01.mp4',help='输入视频的 MP4 文件路径')
    parser.add_argument('--output_dir', type=str, default='./temp', help='输出目录的根路径')
    parser.add_argument('--interval', type=int, default=10, help='每隔N秒提取一帧')
    return parser.parse_args()


def main():
    """主执行函数"""
    args = parse_args()

    # 1. 初始化
    state = init_pipeline(args.input_video, args.output_dir, args.interval)
    if not state:
        return  # 初始化失败则退出

    # 2. 顺序执行所有步骤
    step_extract_audio(state)
    timestamps = step_extract_frames(state)
    deduped_timestamps = step_process_frames(state, timestamps)


    step_ocr_extract(state, deduped_timestamps)
    step_gpt_rewrite_text(state)

    ######### (optional) using image to extract ppt narration #########
    # step_encode_frames(state, deduped_timestamps)
    # step_gpt_rewrite_image(state)

    print("\n==================================================")
    print("🎉 所有处理步骤已全部完成！")
    print(f"最终结果保存在目录: {state['output_dir']}")
    print("==================================================")


if __name__ == "__main__":
    main()
