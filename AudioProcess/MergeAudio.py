import os
import sys
from pydub import AudioSegment

if len(sys.argv) < 2:
    print("用法: python MergeAudio.py <论文目录名，如ClavaDDPM>")
    sys.exit(1)

PAPER_DIR = sys.argv[1]
AUDIO_DIR = f"../tmp/{PAPER_DIR}/tts/"  # 假设每个 section 的音频都在这里
OUTPUT_PATH = f"../tmp/{PAPER_DIR}/tts/full_audio.mp3"

# 获取所有 mp3 文件，按文件名排序
files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.mp3')]
files.sort()

combined = AudioSegment.empty()
for fname in files:
    audio = AudioSegment.from_mp3(os.path.join(AUDIO_DIR, fname))
    combined += audio

combined.export(OUTPUT_PATH, format="mp3")
print(f"合并音频已保存到 {OUTPUT_PATH}")
