from pathlib import Path
import json

def get_video_name(input_path):
    """
    获取视频文件名（不带扩展名）。
    """
    return Path(input_path).stem

def create_directories(output_dir):
    """
    创建保存音频和图片帧的目录。
    """
    audio_dir = output_dir / 'audio'
    frame_dir = output_dir / 'frame'
    audio_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir, frame_dir

def save_timestamps(timestamps, json_path):
    """
    将时间戳列表保存到 JSON 文件中。
    """
    # 将时间戳转换为列表，单位为秒
    timestamps_list = timestamps
    data = {"timestamps": timestamps_list}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)