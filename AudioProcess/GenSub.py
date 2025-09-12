import os
import sys
import whisper
import re

if len(sys.argv) < 2:
    sys.exit(1)

PAPER_DIR = sys.argv[1]
AUDIO_PATH = f"../tmp/{PAPER_DIR}/tts/full_audio.mp3"
SRT_PATH = os.path.join(os.path.dirname(AUDIO_PATH), "output.srt")

model = whisper.load_model("base")
result = model.transcribe(AUDIO_PATH, task="transcribe", language="en")

def sec2timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"



def merge_segments(segments, min_duration=2, max_chars=80):
    merged = []
    buffer = ""
    start = None
    end = None
    for i, seg in enumerate(segments):
        seg_text = seg['text'].strip()
        if start is None:
            start = seg['start']
        if buffer:
            buffer += " "
        buffer += seg_text
        end = seg['end']

        ends_with_punct = bool(re.search(r'[.!?。！？]$', buffer.strip()))

        # 下一个 segment 的内容
        next_seg = segments[i+1]['text'].strip() if i+1 < len(segments) else ""
        # 如果 buffer 已经很长且以标点结尾，直接断开
        if (end - start >= min_duration or len(buffer) >= max_chars):
            if ends_with_punct or len(buffer) >= max_chars:
                merged.append({'start': start, 'end': end, 'text': buffer})
                buffer = ""
                start = None
    # 处理最后一条
    if buffer:
        merged.append({'start': start, 'end': end, 'text': buffer})
    return merged

# 生成 SRT
merged_segments = merge_segments(result["segments"], min_duration=2, max_chars=80)

with open(SRT_PATH, "w", encoding="utf-8") as f:
    for i, seg in enumerate(merged_segments, 1):
        start = sec2timestamp(seg["start"])
        end = sec2timestamp(seg["end"])
        text = seg["text"].strip()
        f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
