import os
import sys
import re
from moviepy import (
    VideoClip,
    TextClip,
    AudioFileClip,
    CompositeVideoClip,
    ColorClip,
)
from moviepy.video.tools.subtitles import SubtitlesClip


if len(sys.argv) < 2:
    sys.exit(1)

PAPER_DIR = sys.argv[1]
AUDIO_PATH = f"../tmp/{PAPER_DIR}/tts/full_audio.mp3"
SRT_PATH = os.path.join(os.path.dirname(AUDIO_PATH), "output.srt")
OUTPUT_PATH = os.path.join(os.path.dirname(AUDIO_PATH), "output.mp4")
VIDEO_SIZE = (1920, 1080)
BG_COLOR = (30, 30, 30)
FONT_SIZE = 88
FONT_COLOR = "yellow"
FONT_NAME = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BORDER_COLOR = "black"
BORDER_WIDTH = 4

# 1. 加载音频并获取总时长
audio_clip = AudioFileClip(AUDIO_PATH)
video_duration = audio_clip.duration

# 2. 创建纯色背景剪辑
background_clip = ColorClip(
    size=VIDEO_SIZE, color=BG_COLOR, duration=video_duration
)

# 3. 创建字幕剪辑
def subtitle_generator(txt):
    return TextClip(
        txt,
        font=FONT_NAME,
        fontsize=FONT_SIZE,
        color=FONT_COLOR,
        stroke_color=BORDER_COLOR,
        stroke_width=BORDER_WIDTH,
        method="caption", # 'caption' 会自动换行
        size=(VIDEO_SIZE[0] * 0.85, None), # 限制字幕宽度
    )

# 使用 SubtitlesClip 创建字幕层
# 这个函数会自动处理时间对齐
subtitles = SubtitlesClip(SRT_PATH, FONT_NAME)

# 4. 组合所有剪辑
subtitles_positioned = subtitles.with_position(("center", "bottom")) #水平中央 竖直底部

final_clip = CompositeVideoClip(
    [background_clip, subtitles_positioned], size=VIDEO_SIZE
)

# 5. 将音频附加到最终的视频剪辑中
final_clip = final_clip.with_audio(audio_clip)

# 6. 写入最终的视频文件
final_clip.write_videofile(
    OUTPUT_PATH,
    codec="libx264", # 推荐的视频编码器
    audio_codec="aac", # 推荐的音频编码器
    temp_audiofile="temp-audio.m4a",
    remove_temp=True,
    fps=24, 
)


audio_clip.close()
final_clip.close()