# -*- coding: utf-8 -*-

import os
import glob
import re
import whisper
import imageio
import numpy as np
from collections import defaultdict
from moviepy import (
    ImageClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    concatenate_audioclips,  # 修正: 添加缺失的导入
    ImageSequenceClip,
    VideoFileClip,
    AudioClip,
    vfx,
    TextClip  # 修正: 为清晰起见添加
)
from moviepy.video.tools.subtitles import SubtitlesClip

# --- 新的辅助函数：使用 OpenCV 和 Pillow 创建字幕 ---
from PIL import Image, ImageDraw, ImageFont

def create_subtitle_clip_opencv(text, duration, clip_size, font_path="Arial.ttf", fontsize=48, txt_color=(255, 255, 255, 255), bg_color=(0, 0, 0, 128)):
    """
    使用Pillow和OpenCV创建一个字幕剪辑，避免使用ImageMagick。

    :param text: 字幕文本
    :param duration: 剪辑时长
    :param clip_size: 视频尺寸 (width, height)
    :param font_path: 字体文件路径。尝试使用系统常见字体。
    :param fontsize: 字体大小
    :param txt_color: 文字颜色 (R, G, B, Alpha)
    :param bg_color: 背景颜色 (R, G, B, Alpha)
    :return: moviepy.ImageClip
    """
    # 尝试找到一个可用的字体
    font_options = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "Arial.ttf", # 在某些系统上 Pillow 可以直接找到
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    
    found_font = font_path
    if not os.path.exists(font_path):
        for font in font_options:
            if os.path.exists(font):
                found_font = font
                print(f"✅ 使用备用字体: {found_font}")
                break
    
    try:
        font = ImageFont.truetype(found_font, fontsize)
    except IOError:
        print(f"❌ 字体 '{found_font}' 加载失败，使用 Pillow 默认字体。")
        font = ImageFont.load_default()

    # 创建一个透明的背景图像
    img = Image.new('RGBA', clip_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # 计算文本尺寸
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # 为了更好的可读性，在文字下面画一个半透明的背景条
    # 你可以根据需要调整 padding
    padding = 10
    margin_bottom = 30  # 距离底部的边距
    
    # 确保文本位置在边界内，但保持统一的字体大小
    text_x = max(padding, min(clip_size[0] - text_width - padding, (clip_size[0] - text_width) / 2))
    text_y = max(padding, min(clip_size[1] - text_height - margin_bottom, clip_size[1] - text_height - margin_bottom))
    
    bg_rect_pos = [
        text_x - padding, 
        text_y - padding,
        text_x + text_width + padding, 
        text_y + text_height + padding
    ]
    # draw.rectangle(bg_rect_pos, fill=bg_color) # 如果需要背景条，取消此行注释

    # 在图像上绘制文字（带描边效果）
    text_pos = (text_x, text_y)
    stroke_width = 2
    stroke_color = "black"
    draw.text((text_pos[0]-stroke_width, text_pos[1]), text, font=font, fill=stroke_color)
    draw.text((text_pos[0]+stroke_width, text_pos[1]), text, font=font, fill=stroke_color)
    draw.text((text_pos[0], text_pos[1]-stroke_width), text, font=font, fill=stroke_color)
    draw.text((text_pos[0], text_pos[1]+stroke_width), text, font=font, fill=stroke_color)
    draw.text(text_pos, text, font=font, fill=txt_color)

    # 将Pillow图像转换为numpy数组，以便moviepy使用
    cv_image = np.array(img)

    # 创建一个 moviepy 剪辑
    subtitle_clip = ImageClip(cv_image).with_duration(duration)
    return subtitle_clip

# --- 辅助函数：从之前的脚本中提取并优化 ---

def _generate_srt_from_audio(audio_path: str, srt_path: str):
    """
    使用 Whisper 模型为给定的音频文件生成 SRT 字幕。

    :param audio_path: 输入的音频文件路径。
    :param srt_path: 输出的 SRT 文件路径。
    """
    print(f"--- 开始生成字幕，音频源: {audio_path} ---")
    try:
        # 加载模型 (base 模型在性能和速度上是个不错的起点)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, task="transcribe", language="en", fp16=False)

        def sec2timestamp(sec):
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int((sec - int(sec)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        # 这个函数将短的字幕片段合并成更易读的句子
        def merge_segments(segments, min_duration=2.0, max_chars=80):
            merged = []
            buffer = ""
            start_time = None
            end_time = None
            for i, seg in enumerate(segments):
                seg_text = seg['text'].strip()
                if not seg_text:
                    continue
                if start_time is None:
                    start_time = seg['start']

                buffer = (buffer + " " + seg_text) if buffer else seg_text
                end_time = seg['end']

                ends_with_punct = bool(re.search(r'[.!?。！？]$', buffer.strip()))

                if (end_time - start_time >= min_duration) or (len(buffer) >= max_chars):
                    if ends_with_punct or len(buffer) >= max_chars:
                        # 确保结束时间精确反映语音结束
                        merged.append({'start': start_time, 'end': end_time, 'text': buffer})
                        buffer = ""
                        start_time = None

            if buffer and start_time is not None:
                merged.append({'start': start_time, 'end': end_time, 'text': buffer})
            return merged

        merged_segments_data = merge_segments(result["segments"])

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(merged_segments_data, 1):
                start = sec2timestamp(seg["start"])
                end = sec2timestamp(seg["end"])
                text = seg["text"].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

        print(f"--- 字幕文件已成功保存到: {srt_path} ---")
        return True
    except Exception as e:
        print(f"生成字幕时发生严重错误: {e}")
        return False


# --- 主函数：整合了图片、音频、字幕和GIF ---

def create_presentation_video(
        args,
        image_dir: str,
        tts_audio_files: dict,
        page_to_section_map: dict,
        output_video_path: str,
        overlay_gif_path: str = None,
        manim_video_path: str = None,
        font_path: str = "DejaVu-Sans-Mono",  # 使用一个更通用的字体名称
        fps: int = 24
):
    """
    将一系列图片、对应的音频和字幕合成为一个带解说的视频。

    :param image_dir: 包含图片文件（如 0.png, 1.png ...）的目录。
    :param tts_audio_files: 字典，键是章节名，值是对应的音频文件路径。
    :param page_to_section_map: 字典，键是图片页码（整数），值是对应的章节名。
    :param output_video_path: 输出视频的文件路径。
    :param fps: 视频的帧率。
    :param overlay_gif_path: (可选) 要叠加在右下角的GIF文件路径。
    :param font_path: (可选) 用于字幕的字体文件路径或字体名称。
    """
    # ------------------ 步骤 1: 准备工作和数据校验 ------------------
    print("==== 开始视频生成流程 ====")

    # 获取图片文件并排序
    try:
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        num_pages = len(image_files)
        print(f"找到 {num_pages} 张图片。")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: 无法处理图片目录 '{image_dir}'. {e}")
        return

    # ------------------ 新增步骤: Manim视频插入点计算 ------------------

    manim_insert_page = -1
    should_use_manim = args.use_manim and manim_video_path and os.path.exists(manim_video_path)
    if should_use_manim:
        # 智能插入点计算：优先寻找 Implementation 章节，否则插入到最后一页之前
        implementation_pages = [p for p, s in page_to_section_map.items() if s and "implementation" in s.lower()]
        if implementation_pages:
            # 找到 Implementation 章节，插入到该章节之后
            manim_insert_page = max(implementation_pages) + 1
            print(f"🔍 找到 Implementation 章节，Manim视频将插入在第 {manim_insert_page} 页")
        else:
            # 未找到 Implementation 章节，插入到最后一页之前
            manim_insert_page = max(0, num_pages - 1)  # 确保不为负数
            print(f"⚠️ 未找到 Implementation 章节，Manim视频将插入在最后一页之前（第 {manim_insert_page} 页）")
        
        # 安全检查插入点是否合理
        if manim_insert_page < 0 or manim_insert_page > num_pages:
            print(f"⚠️ 插入点 {manim_insert_page} 超出页面范围 [0, {num_pages}]，跳过 Manim 插入。")
            should_use_manim = False
    
    elif args.use_manim:
        print(f"警告: Manim视频路径 '{manim_video_path}' 无效或未提供，将不插入视频。")
        should_use_manim = False

    # ------------------ 步骤 2: 计算每张幻灯片的时长并合并音频 ------------------
    print("\n==== 步骤 2: 计算时长并合并音频 ====")
    slide_durations = [0] * num_pages
    tts_audio_clips = []
    section_usage_count = defaultdict(int)

    for page_num in range(num_pages):
        section_name = page_to_section_map.get(page_num)
        if section_name:
            section_usage_count[section_name] += 1

    # 修复音频处理逻辑
    section_audio_clips = {}  # 缓存每个section的音频文件
    section_audio_positions = defaultdict(int)  # 跟踪每个section的音频位置
    
    for i in range(num_pages):
        section_name = page_to_section_map.get(i)
        if not section_name or section_name not in tts_audio_files:
            print(f"警告: 第 {i} 页没有找到对应的章节或音频，将使用默认时长 3 秒。")
            slide_durations[i] = 3
            # 添加一个短静音片段以保持同步
            tts_audio_clips.append(AudioClip(lambda t: 0, duration=3, fps=44100))
            continue

        audio_path = tts_audio_files[section_name]
        try:
            # 缓存音频文件，避免重复加载
            if section_name not in section_audio_clips:
                section_audio_clips[section_name] = AudioFileClip(audio_path)
            
            audio_clip = section_audio_clips[section_name]
            total_duration = audio_clip.duration
            section_count = section_usage_count[section_name]
            
            # 计算当前页面应该使用的音频片段
            segment_duration = total_duration / section_count
            start_time = section_audio_positions[section_name] * segment_duration
            end_time = start_time + segment_duration
            
            # 确保不超出音频长度
            if end_time > total_duration:
                end_time = total_duration
            
            # 创建音频片段
            audio_segment = audio_clip.subclipped(start_time, end_time)
            actual_duration = audio_segment.duration
            slide_durations[i] = actual_duration
            tts_audio_clips.append(audio_segment)
            
            # 更新位置计数器
            section_audio_positions[section_name] += 1
            
            print(f"页面 {i} ({section_name}): 分配时长 {actual_duration:.2f}s (片段 {section_audio_positions[section_name]}/{section_count})")
        except Exception as e:
            print(f"错误: 无法加载音频文件 {audio_path}。错误: {e}")
            slide_durations[i] = 3
            tts_audio_clips.append(AudioClip(lambda t: 0, duration=3, fps=44100))

    # 合并TTS音频 - 使用pydub方法（更可靠）
    print("🔊 使用pydub合并音频文件...")
    try:
        from pydub import AudioSegment
        
        # 创建临时目录用于音频处理
        temp_audio_dir = "./temp_video_assets"
        os.makedirs(temp_audio_dir, exist_ok=True)
        
        # 使用pydub合并音频
        combined_audio = AudioSegment.empty()
        for i, audio_clip in enumerate(tts_audio_clips):
            # 将MoviePy音频剪辑转换为临时文件
            temp_audio_path = os.path.join(temp_audio_dir, f"temp_audio_{i}.mp3")
            # 修复：新版本MoviePy的write_audiofile参数变化
            audio_clip.write_audiofile(temp_audio_path, logger=None)
            
            # 使用pydub加载并合并
            segment = AudioSegment.from_mp3(temp_audio_path)
            combined_audio += segment
            
            # 清理临时文件
            os.remove(temp_audio_path)
        
        # 如果需要，插入Manim的静音片段
        #lbx modified
        # if should_use_manim:
        #     print("正在将静音片段插入音频轨道以匹配Manim视频...")
        #     manim_clip_for_audio = VideoFileClip(manim_video_path)
        #     manim_duration = manim_clip_for_audio.duration
        #     silent_segment = AudioSegment.silent(duration=int(manim_duration * 1000))  # pydub使用毫秒

        #     # 计算音频插入点
        #     audio_insertion_time = sum(slide_durations[:manim_insert_page])
        #     print(f"音频插入点: {audio_insertion_time:.2f}s")

        #     # 分割并重组音频
        #     insertion_ms = int(audio_insertion_time * 1000)
        #     if insertion_ms < len(combined_audio):
        #         part1 = combined_audio[:insertion_ms]
        #         part2 = combined_audio[insertion_ms:]
        #         combined_audio = part1 + silent_segment + part2
        #     else:
        #         # 如果插入点超出音频长度，直接添加静音
        #         combined_audio = combined_audio + silent_segment
        #####start

        if should_use_manim:
            print("🔊 正在处理Manim视频的音轨...")

            # 修正第一步：无论如何，先加载Manim视频剪辑以获取信息
            manim_clip_for_audio = VideoFileClip(manim_video_path)
            manim_duration_ms = int(manim_clip_for_audio.duration * 1000)

            # 现在检查我们自定义的音轨
            manim_audio_path = tts_audio_files.get("Manim_Animation", tts_audio_files.get("manim_narration1")) # 尝试两个可能的键名

            if manim_audio_path and os.path.exists(manim_audio_path):
                print(f"✅ 找到Manim音轨: {manim_audio_path}")
                manim_audio_segment = AudioSegment.from_mp3(manim_audio_path)

                # 可选：如果音频时长与视频时长差异过大，可以截断或填充
                # 这里我们简单地截断音频以匹配视频时长
                if len(manim_audio_segment) > manim_duration_ms:
                    manim_audio_segment = manim_audio_segment[:manim_duration_ms]

            else:
                print("⚠️ 未找到自定义Manim音轨，将使用静音。")
                manim_audio_segment = AudioSegment.silent(duration=manim_duration_ms)

            # 修正第二步：在使用完剪辑后，及时关闭
            manim_clip_for_audio.close()

            # 计算音频插入点
            audio_insertion_time = sum(slide_durations[:manim_insert_page])
            print(f"音频插入点: {audio_insertion_time:.2f}s")

            # 分割并重组音频
            insertion_ms = int(audio_insertion_time * 1000)
            if insertion_ms < len(combined_audio):
                part1 = combined_audio[:insertion_ms]
                part2 = combined_audio[insertion_ms:]
                combined_audio = part1 + manim_audio_segment + part2
            else:
                combined_audio = combined_audio + manim_audio_segment
        #####end
            manim_clip_for_audio.close()
        
        # 将合并后的音频保存为临时文件
        temp_combined_audio_path = os.path.join(temp_audio_dir, "combined_audio.mp3")
        combined_audio.export(temp_combined_audio_path, format="mp3")
        
        # 从临时文件创建MoviePy音频剪辑
        full_audio_track = AudioFileClip(temp_combined_audio_path)
        print(f"✅ 音频合并成功，总时长: {full_audio_track.duration:.2f}s")
        
    except Exception as e:
        print(f"❌ pydub音频合并失败: {e}")
        print("🔄 回退到MoviePy音频合并方法...")
        
        # 回退到原来的MoviePy方法
        concatenated_tts_audio = concatenate_audioclips(tts_audio_clips)
        
        # 如果需要，插入Manim的静音片段
        if should_use_manim:
            print("正在将静音片段插入音频轨道以匹配Manim视频...")
            manim_clip_for_audio = VideoFileClip(manim_video_path)
            manim_duration = manim_clip_for_audio.duration
            silent_audio = AudioClip(lambda t: 0, duration=manim_duration, fps=44100)

            # 计算音频插入点
            audio_insertion_time = sum(slide_durations[:manim_insert_page])
            print(f"音频插入点: {audio_insertion_time:.2f}s")

            # 分割并重组音频
            if audio_insertion_time < concatenated_tts_audio.duration:
                # 修复：新版本MoviePy使用subclipped而不是subclip
                audio_part1 = concatenated_tts_audio.subclipped(0, audio_insertion_time)
                audio_part2 = concatenated_tts_audio.subclipped(audio_insertion_time)
                full_audio_track = concatenate_audioclips([audio_part1, silent_audio, audio_part2])
            else:
                full_audio_track = concatenate_audioclips([concatenated_tts_audio, silent_audio])
            
            manim_clip_for_audio.close()
        else:
            full_audio_track = concatenated_tts_audio

    total_duration = full_audio_track.duration
    print(f"音频轨道构建完成，总时长: {total_duration:.2f}s")
    
    # 验证音频是否有效
    if total_duration <= 0:
        print("❌ 错误: 音频轨道时长为0，将使用静音")
        full_audio_track = AudioClip(lambda t: 0, duration=sum(slide_durations), fps=44100)
    else:
        print(f"✅ 音频轨道验证通过，总时长: {total_duration:.2f}s")
        
        # 额外验证：检查音频是否有实际内容（不是完全静音）
        try:
            # 尝试获取音频的振幅信息
            test_frame = full_audio_track.get_frame(0.1)  # 获取0.1秒处的帧
            if isinstance(test_frame, np.ndarray):
                max_amplitude = np.max(np.abs(test_frame))
                print(f"✅ 音频振幅检查: 最大振幅 = {max_amplitude:.6f}")
                if max_amplitude < 0.001:  # 如果振幅太小，可能是静音
                    print("⚠️ 警告: 音频振幅很小，可能是静音")
            else:
                print("✅ 音频帧检查通过")
        except Exception as e:
            print(f"⚠️ 音频振幅检查失败: {e}")

    # ------------------ 步骤 3: 生成字幕文件 ------------------
    print("\n==== 步骤 3: 生成字幕 ====")
    temp_dir = "./temp_video_assets"
    os.makedirs(temp_dir, exist_ok=True)
    temp_audio_path = os.path.join(temp_dir, "full_audio.mp3")
    temp_srt_path = os.path.join(temp_dir, "subtitles.srt")

    # 添加音频调试信息
    print(f"🔊 音频调试信息:")
    print(f"   - 音频轨道时长: {full_audio_track.duration:.2f}s")
    print(f"   - 音频采样率: {full_audio_track.fps if hasattr(full_audio_track, 'fps') else 'N/A'}")
    print(f"   - 临时音频文件: {temp_audio_path}")
    
    # 检查原始TTS音频文件
    print(f"🔍 检查原始TTS音频文件:")
    for section_name, audio_path in tts_audio_files.items():
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"   - {section_name}: {file_size} bytes")
            
            # 尝试加载并检查音频
        try:
            test_audio = AudioFileClip(audio_path)
            # 修复：通过传递一个明确的时间段来规避 to_soundarray 的 bug
            audio_array = test_audio.get_frame(t=0.1) # 获取一个样本帧
            # 我们不再尝试读取整个数组，而是直接检查时长
            print(f"     - 时长: {test_audio.duration:.2f}s (振幅检查已跳过)")
            test_audio.close()
        except Exception as e:
            print(f"     - 无法加载音频: {e}")
            try:
                test_audio = AudioFileClip(audio_path)
                print(f"     - 时长: {test_audio.duration:.2f}s (无法获取振幅)")
                test_audio.close()
            except:
                print(f"     - 完全无法加载音频文件")
        else:
            print(f"   - {section_name}: 文件不存在")

    try:
        # 检查音频是否有效（不是静音）
        try:
            # 我们在这里只做最简单的检查，因为主要问题是 to_soundarray
            if full_audio_track.duration > 0:
                print(f"🔊 音频轨道有效，时长: {full_audio_track.duration:.2f}s")
                full_audio_track.write_audiofile(temp_audio_path)
                print(f"✅ 音频文件写入成功: {temp_audio_path}")
            else:
                print("❌ 音频轨道时长为0")
                temp_audio_path = None
        except Exception as e:
            print(f"❌ 音频文件写入失败: {e}")
            temp_audio_path = None
            
        # 验证音频文件
        if temp_audio_path and os.path.exists(temp_audio_path):
            file_size = os.path.getsize(temp_audio_path)
            print(f"✅ 音频文件大小: {file_size} bytes")
            
            # 检查文件是否真的包含音频数据
            if file_size < 1000:  # 小于1KB可能是空文件
                print("⚠️ 警告: 音频文件太小，可能没有有效音频数据")
            else:
                print(f"✅ 音频文件大小正常: {file_size} bytes")
        else:
            print("❌ 音频文件写入失败")
            
    except Exception as e:
        print(f"❌ 音频文件写入失败: {e}")
        temp_audio_path = None
    
    if temp_audio_path and os.path.exists(temp_audio_path):
        # 测试音频文件是否有声音
        try:
            import subprocess
            # 使用 ffprobe 检查音频文件
            result = subprocess.run(['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', '-of', 'csv=p=0', temp_audio_path], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                duration = float(result.stdout.strip())
                print(f"✅ 音频文件验证成功，时长: {duration:.2f}s")
                
                # 检查音频文件大小
                file_size = os.path.getsize(temp_audio_path)
                if file_size > 10000:  # 大于10KB
                    print(f"✅ 音频文件大小正常: {file_size} bytes")
                    
                    # 额外验证：尝试加载音频文件确保它是有效的
                    try:
                        test_audio = AudioFileClip(temp_audio_path)
                        print(f"✅ 音频文件可以正常加载，时长: {test_audio.duration:.2f}s")
                        test_audio.close()
                        
                        if not _generate_srt_from_audio(temp_audio_path, temp_srt_path):
                            print("字幕生成失败，视频将不包含字幕。")
                            temp_srt_path = None
                    except Exception as e:
                        print(f"❌ 音频文件加载失败: {e}")
                        temp_srt_path = None
                else:
                    print(f"⚠️ 音频文件太小: {file_size} bytes，可能没有有效音频")
                    temp_srt_path = None
            else:
                print("❌ 音频文件验证失败")
                temp_srt_path = None
        except Exception as e:
            print(f"❌ 音频文件验证出错: {e}")
            temp_srt_path = None
    else:
        print("❌ 无法生成字幕：音频文件不存在")
        temp_srt_path = None

    # ------------------ 步骤 4: 从图片创建视频剪辑 ------------------
    print("\n==== 步骤 4: 创建主视频轨道 ====")
    video_segments = []
    target_size = None  # 用于存储幻灯片的统一尺寸
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        clip = ImageClip(image_path).with_duration(slide_durations[i])
        if target_size is None:
            target_size = clip.size
            print(f"所有视频剪辑的目标尺寸设置为: {target_size}")
        video_segments.append(clip)

    if should_use_manim:
        print(f"正在加载Manim视频: {manim_video_path}")
        manim_clip = VideoFileClip(manim_video_path)
        if manim_clip.size != target_size and target_size is not None:
            print(f"正在将Manim视频从 {manim_clip.size} 调整到 {target_size}")
            manim_clip = manim_clip.resized(target_size)
        video_segments.insert(manim_insert_page, manim_clip)

    main_clip = concatenate_videoclips(video_segments, method="compose")
    main_clip.fps = fps
    
    # 确保音频和视频时长匹配
    video_duration = main_clip.duration
    audio_duration = full_audio_track.duration
    
    print(f"视频时长: {video_duration:.2f}s, 音频时长: {audio_duration:.2f}s")
    
    # 如果时长不匹配，进行调整
    if abs(video_duration - audio_duration) > 0.1:  # 允许0.1秒的误差
        print(f"⚠️ 音频和视频时长不匹配，进行调整...")
        if video_duration > audio_duration:
            # 视频更长，延长音频
            padding_duration = video_duration - audio_duration
            padding_audio = AudioClip(lambda t: 0, duration=padding_duration, fps=44100)
            full_audio_track = concatenate_audioclips([full_audio_track, padding_audio])
        else:
            # 音频更长，截断音频
            full_audio_track = full_audio_track.subclipped(0, video_duration)
    
    #main_clip = main_clip.set_audio(full_audio_track)  # 将完整音轨附加到视频
    
    

    # ------------------ 步骤 5: 叠加字幕和GIF ------------------
    print("\n==== 步骤 5: 叠加字幕和可选的GIF ====")
    clips_to_composite = [main_clip]

    # --- 新的字幕添加逻辑 (Plan B) ---
    if temp_srt_path and os.path.exists(temp_srt_path):
        print("正在使用 OpenCV 和 Pillow 添加字幕...")
        try:
            from moviepy.video.tools.subtitles import SubtitlesClip
            from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
            import pysrt # 需要安装: pip install pysrt

            subs = pysrt.open(temp_srt_path)
            subtitle_clips = []
            fade_duration = 0.2  # 减少淡出时间，避免重叠
            min_gap = 0.1  # 字幕之间的最小间隔
            
            for i, sub in enumerate(subs):
                start_time = sub.start.to_time().hour * 3600 + sub.start.to_time().minute * 60 + sub.start.to_time().second + sub.start.to_time().microsecond / 1e6
                end_time = sub.end.to_time().hour * 3600 + sub.end.to_time().minute * 60 + sub.end.to_time().second + sub.end.to_time().microsecond / 1e6
                
                # 确保字幕在语音结束后立即消失，不给下一句留重叠时间
                display_duration = (end_time - start_time)
                
                # 如果下一句字幕开始时间太近，提前结束当前字幕
                if i < len(subs) - 1:
                    next_start = subs[i+1].start.to_time().hour * 3600 + subs[i+1].start.to_time().minute * 60 + subs[i+1].start.to_time().second + subs[i+1].start.to_time().microsecond / 1e6
                    if next_start - end_time < min_gap:
                        display_duration = (end_time - start_time) - (min_gap - (next_start - end_time))
                        display_duration = max(display_duration, 0.1)  # 确保至少显示0.1秒
                
                # 使用我们的新函数创建剪辑，使用统一的字体大小
                text_clip = create_subtitle_clip_opencv(
                    sub.text, 
                    display_duration, 
                    main_clip.size,
                    fontsize=36  # 统一使用36号字体
                )
                
                # 添加淡出效果：在语音结束前开始淡出，确保不重叠
                fade_start_time = max(0, display_duration - fade_duration)
                try:
                    # 使用fade方法，在字幕结束前开始淡出
                    text_clip_with_fade = text_clip.fade(0, fade_duration)
                except Exception as e:
                    try:
                        # 尝试使用fadeout方法
                        text_clip_with_fade = text_clip.fadeout(fade_duration)
                    except AttributeError:
                        try:
                            # 尝试使用vfx.fadeout方法
                            text_clip_with_fade = vfx.fadeout(text_clip, fade_duration)
                        except AttributeError:
                            # 如果所有方法都不可用，直接使用原剪辑（无淡出效果）
                            print(f"⚠️ 警告: 无法应用淡出效果，字幕将直接消失 (错误: {e})")
                            text_clip_with_fade = text_clip
                
                # 设置剪辑的开始时间并添加到列表
                subtitle_clips.append(text_clip_with_fade.with_start(start_time))

            # 将所有字幕剪辑和主视频合成为一个
            clips_to_composite.extend(subtitle_clips)
            print("✅ 字幕添加成功 (使用OpenCV方案，带淡出效果)")

        except Exception as e:
            print(f"❌ 使用OpenCV方案添加字幕时失败: {e}")

    # 添加GIF
    if overlay_gif_path:
        print(f"正在添加GIF: {overlay_gif_path}")
        try:
            # 修正: 使用 imageio 正确读取GIF文件
            gif_reader = imageio.get_reader(overlay_gif_path)
            # 从GIF元数据获取fps，如果失败则使用视频的fps
            gif_fps = gif_reader.get_meta_data().get('fps', fps)
            gif_frames = [frame for frame in gif_reader]

            # 从帧列表创建剪辑
            gif_clip = ImageSequenceClip(gif_frames, fps=gif_fps)

            resized_gif = gif_clip.resized(height=main_clip.h * 0.25)  # GIF高度为视频的25%
            # 修复：新版本MoviePy中fx方法可能不可用，直接使用loop方法
            try:
                looped_gif = resized_gif.fx(vfx.loop, duration=main_clip.duration)
            except AttributeError:
                # 如果fx不可用，手动创建循环
                num_loops = int(main_clip.duration / resized_gif.duration) + 1
                looped_gif = concatenate_videoclips([resized_gif] * num_loops)
                looped_gif = looped_gif.subclipped(0, main_clip.duration)
            positioned_gif = looped_gif.with_position(("right", "bottom"))
            clips_to_composite.append(positioned_gif)
        except Exception as e:
            print(f"警告: 无法加载或处理GIF '{overlay_gif_path}'. 错误: {e}")

    # 先给主视频剪辑附加音频
    print("🔊 正在给主视频剪辑附加音频...")
    main_clip_with_audio = main_clip.with_audio(full_audio_track)
    
    # 然后创建最终的合成剪辑
    final_clip = CompositeVideoClip([main_clip_with_audio] + clips_to_composite[1:])  # 保持主视频的音频，其他元素叠加
    # final_clip.set_duration(main_clip.duration)

    # ------------------ 步骤 6: 写入最终视频文件 ------------------
    print("\n==== 步骤 6: 正在写入最终视频文件... (这可能需要一些时间) ====")
    print(output_video_path, fps)
    
    # 确保音频正确附加到最终剪辑
    print("🔊 正在验证最终剪辑的音频...")
    final_clip_with_audio = final_clip  # 音频已经在主视频中
    
    # vvvv 在这里添加最终调试代码 vvvv
    print("--- FINAL CLIP DEBUG INFO ---")
    print(f"Final clip duration = {final_clip_with_audio.duration}")
    print(f"Final clip size (width, height) = {final_clip_with_audio.size}")
    print(f"Has audio? {'Yes' if final_clip_with_audio.audio is not None else 'No'}")
    if final_clip_with_audio.audio is not None:
        print(f"Audio duration = {final_clip_with_audio.audio.duration}")
        print(f"Audio fps = {final_clip_with_audio.audio.fps if hasattr(final_clip_with_audio.audio, 'fps') else 'N/A'}")
        
       
    
    print('fps:', fps)
    
    # 使用更明确的音频参数
    try:
        # 修复：新版本MoviePy的write_videofile参数变化
        final_clip_with_audio.write_videofile(
            output_video_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile='temp-audio.m4a', 
            remove_temp=True,
            threads=4, 
            logger='bar',
            ffmpeg_params=["-strict", "-2"]  # 添加ffmpeg参数以确保音频正确处理
        )
        print(f"\n视频成功创建！已保存至：{output_video_path}")
    except Exception as e:
        print(f"❌ 视频写入失败: {e}")
        print("🔄 尝试备用方法：直接使用ffmpeg合并视频和音频...")
        
        # 备用方法：先生成无音频视频，然后用ffmpeg合并
        try:
            # 生成无音频视频
            temp_video_path = output_video_path.replace('.mp4', '_temp.mp4')
            # 修复：新版本MoviePy的write_videofile参数变化
            final_clip_with_audio.write_videofile(
                temp_video_path,
                codec="libx264",
                audio=False,  # 不包含音频
                temp_audiofile=None,
                remove_temp=True,
                threads=4, 
                logger='bar'
            )
            
            # 使用ffmpeg合并视频和音频
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', temp_audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', '-2',
                output_video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ 备用方法成功！视频已保存至：{output_video_path}")
                # 清理临时文件
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            else:
                print(f"❌ 备用方法也失败了: {result.stderr}")
        except Exception as backup_e:
            print(f"❌ 备用方法也失败了: {backup_e}")

    # print("清理临时文件...")
    # full_audio_track.close()
    # for clip in tts_audio_clips:
    #     clip.close()
    # if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
    # if os.path.exists(temp_srt_path): os.remove(temp_srt_path)
    # if os.path.exists(temp_dir): os.rmdir(temp_dir)


if __name__ == '__main__':
    # =================================================================
    # --- 使用示例 ---
    # 假设你的图片在 './data/my_poster/images/' 目录下，命名为 0.png, 1.png ...
    # =================================================================

    # 1. 定义你的数据结构
    tts_audio_files_example = {
        "Poster Title & Author": "./contents/tts/section_01_Poster Title & Author.mp3",
        "Abstract": "./contents/tts/section_02_Abstract.mp3",
        "Introduction": "./contents/tts/section_03_Introduction.mp3",
        "Related Work": "./contents/tts/section_04_Related Work.mp3",
        "Methodology": "./contents/tts/section_05_Methodology.mp3",
        "Experimental Settings": "./contents/tts/section_06_Experimental Settings.mp3",
        "Experimental Results": "./contents/tts/section_07_Experimental Results.mp3",
        "Conclusion": "./contents/tts/section_08_Conclusion.mp3"
    }

    page_to_section_map_example = {
        0: 'Poster Title & Author',
        1: 'Abstract',
        2: 'Introduction',
        3: 'Related Work',
        4: 'Methodology',  # Methodology的第一页
        5: 'Methodology',  # Methodology的第二页
        6: 'Experimental Settings',
        7: 'Experimental Results',
        8: 'Conclusion'
    }

    # 2. 设置路径
    poster_name = "MyAwesomePoster"  # 定义你的项目名
    image_directory = f'./data/{poster_name}/images'
    gif_file_path = f'./data/{poster_name}/kq.gif'  # 如果没有GIF，设为 None
    output_file_path = f'./output/{poster_name}_presentation.mp4'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # --- 模拟创建一些文件用于测试 ---
    print("--- 正在创建用于测试的虚拟文件 ---")
    os.makedirs(image_directory, exist_ok=True)
    for i in range(len(page_to_section_map_example)):
        # 创建空白图片
        from PIL import Image

        img = Image.new('RGB', (1920, 1080), color='darkblue')
        img.save(os.path.join(image_directory, f'{i}.png'))

    os.makedirs("./contents/tts", exist_ok=True)
    for section, path in tts_audio_files_example.items():
        # 创建静音音频
        from pydub import AudioSegment

        silence = AudioSegment.silent(duration=5000 if "Methodology" not in section else 10000)  # 给方法论更长的时间
        silence.export(path, format="mp3")
    print("--- 虚拟文件创建完毕 ---\n")
    # --- 模拟文件创建结束 ---

    # 3. 调用主函数
    create_presentation_video(
        image_dir=image_directory,
        tts_audio_files=tts_audio_files_example,
        page_to_section_map=page_to_section_map_example,
        output_video_path=output_file_path,
        overlay_gif_path=None  # 暂时禁用GIF测试
        # overlay_gif_path=gif_file_path # 如果你有GIF文件，请取消此行注释
    )

