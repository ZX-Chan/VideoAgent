import os
import json
import webvtt

# --- 配置您的文件夹路径 ---
SOURCE_DIR = '/data/ssd/Data/Pictory/generated_narration_vtt'
DESTINATION_DIR = '/data/ssd/Data/Pictory/generated_narration'


# -------------------------

def convert_vtt_to_custom_json(vtt_file_path):
    """
    读取 VTT 文件并将其转换为以时间段为 key 的 JSON 对象。

    Args:
        vtt_file_path (str): VTT 文件的路径。

    Returns:
        str: JSON 格式的字符串，如果失败则返回 None。
    """
    narration_dict = {}
    try:
        # 使用 webvtt-py 库读取文件
        for caption in webvtt.read(vtt_file_path):
            # 将开始和结束时间格式化为 key
            time_key = f"{caption.start} --> {caption.end}"
            # 清理字幕文本，合并多行文本为一个字符串
            text_value = caption.text.strip().replace('\n', ' ')
            narration_dict[time_key] = text_value

        # 将字典转换为格式化的 JSON 字符串
        return json.dumps(narration_dict, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"  [错误] 处理文件 '{os.path.basename(vtt_file_path)}' 时出错: {e}")
        return None


def main():
    """
    主函数，执行所有文件的转换。
    """
    # 1. 检查源文件夹是否存在
    if not os.path.isdir(SOURCE_DIR):
        print(f"错误：源文件夹 '{SOURCE_DIR}' 不存在。请检查路径。")
        return

    # 2. 检查并创建目标文件夹
    if not os.path.exists(DESTINATION_DIR):
        print(f"目标文件夹 '{DESTINATION_DIR}' 不存在，正在创建...")
        os.makedirs(DESTINATION_DIR)
        print(f"文件夹已创建。")

    # 3. 查找所有 .vtt 文件
    vtt_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.vtt')]

    if not vtt_files:
        print(f"在 '{SOURCE_DIR}' 中没有找到任何 .vtt 文件。")
        return

    print(f"找到 {len(vtt_files)} 个 .vtt 文件，开始转换...")

    success_count = 0
    error_count = 0

    # 4. 遍历并转换每个文件
    for vtt_filename in vtt_files:
        full_vtt_path = os.path.join(SOURCE_DIR, vtt_filename)
        print(f"正在处理: {vtt_filename}")

        # 转换 VTT 到 JSON 字符串
        json_output = convert_vtt_to_custom_json(full_vtt_path)

        if json_output:
            # 构建输出文件名
            base_name = os.path.splitext(vtt_filename)[0]
            output_filename = f"{base_name}_narration.json"
            full_output_path = os.path.join(DESTINATION_DIR, output_filename)

            # 写入文件
            try:
                with open(full_output_path, 'w', encoding='utf-8') as f:
                    f.write(json_output)
                success_count += 1
            except IOError as e:
                print(f"  [错误] 无法写入文件 '{output_filename}': {e}")
                error_count += 1
        else:
            error_count += 1

    print("\n--- 转换完成 ---")
    print(f"成功转换: {success_count} 个文件")
    if error_count > 0:
        print(f"失败: {error_count} 个文件")
    print(f"所有 JSON 文件已保存在: '{DESTINATION_DIR}'")


if __name__ == "__main__":
    main()