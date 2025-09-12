#!/bin/bash

# 定义输入和输出目录
INPUT_DIR="/data/ssd/Data/LunWenShuo/videos"
OUTPUT_DIR="/data/ssd/Data/LunWenShuo/processed"
INTERVAL=5

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 查找所有 .mp4 文件并循环处理
for video_file in "$INPUT_DIR"/*.mp4; do
  # 检查文件是否存在，以防目录为空
  if [ -f "$video_file" ]; then
    # 获取文件名（不含路径）
    filename=$(basename -- "$video_file")

    # 打印正在处理的文件信息
    echo "正在处理: $filename"

    # 调用 Python 脚本
    python generate_video_summary.py \
      --input_video "$video_file" \
      --output_dir "$OUTPUT_DIR" \
      --interval $INTERVAL

    echo "处理完成: $filename"
    echo "---------------------------------"
  fi
done

echo "所有视频处理完毕！"