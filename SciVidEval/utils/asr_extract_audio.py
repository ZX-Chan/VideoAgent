import json
import subprocess

from funasr import AutoModel

# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                  vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                  punc_model="ct-punc-c", punc_model_revision="v2.0.4")



def get_single_timeslice(texts, times):
    """
    【备选方案】将整个识别结果作为一个时间片。

    :param texts: str, ASR 得到的完整文本。
    :param times: list, 每个字对应的时间戳。
    :return: list, 只包含一个句子的列表。
    """
    if not texts or not times:
        return []

    start_time = times[0][0]
    end_time = times[-1][1]

    return [{
        "text": texts,
        "start": start_time,
        "end": end_time
    }]

def milliseconds_to_hhmmss(milliseconds):
    """将给定的毫秒数转换为 HH:MM:SS.mmm 格式"""
    total_seconds = milliseconds // 1000
    ms = milliseconds % 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
def extract_audio(input_path, output_wav):
    """
    使用 ffmpeg 提取音频并保存为 WAV 格式。
    """
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-vn',  # 不处理视频部分
        '-acodec', 'pcm_s16le',  # PCM 16位小端格式
        '-ar', '44100',  # 采样率 44100 Hz
        '-ac', '2',  # 双声道
        '-y',  # 覆盖输出文件而不提示
        str(output_wav)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"ffmpeg 提取音频错误: {result.stderr}")


def audio_to_text(input_path, use_segmentation=True):

    res = model.generate(input=f"{input_path}",
                         batch_size_s=300, hotword='魔搭')

    texts = res[0]['text']
    times = res[0]['timestamp']


    segments = get_single_timeslice(texts, times)

    result = {}
    for seg in segments:
        start_str = milliseconds_to_hhmmss(seg['start'])
        end_str = milliseconds_to_hhmmss(seg['end'])
        result_key = f"{start_str}-{end_str}"
        result[result_key] = seg['text']


    output_path = input_path.replace('.wav', '_asr.json').replace('/audio', '')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    return result, output_path

# if __name__ == '__main__':
#     audio_to_text('./temp/1_2_4/audio/1_2_4.wav')