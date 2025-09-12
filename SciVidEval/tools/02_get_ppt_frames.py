import os
import subprocess
import tempfile
from pathlib import Path
from pdf2image import convert_from_path


# 简化路径操作，使用 pathlib
def pexists(path):
    return Path(path).exists()


def pjoin(*args):
    return str(Path(*args))


def ppt_to_multi_images(file: str, output_dir: str, dpi=150, output_type='jpg'):
    """
    Converts a single PPTX file to multiple image files, one for each slide.

    Args:
        file (str): Path to the input PPTX file.
        output_dir (str): Directory to save the output images.
        dpi (int): The DPI for the output images.
        output_type (str): The format for the output images ('png' or 'jpg').
    """
    assert pexists(file), f"File {file} does not exist"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # 使用 soffice 将 PPTX 转换为 PDF
        # `--headless`: 无界面模式
        # `--convert-to pdf`: 指定转换目标格式
        # `--outdir`: 指定输出目录
        command_list = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            file,
            "--outdir",
            temp_dir,
        ]
        # subprocess.run 会等待命令执行完毕
        subprocess.run(command_list, check=True, stdout=subprocess.DEVNULL)

        # 查找临时目录中的 PDF 文件
        pdf_file = None
        for f in os.listdir(temp_dir):
            if f.endswith(".pdf"):
                pdf_file = pjoin(temp_dir, f)
                break

        if not pdf_file:
            raise RuntimeError("No PDF file was created in the temporary directory", file)

        # 使用 pdf2image 将 PDF 转换为图片
        # `convert_from_path` 会返回一个 PIL Image 对象的列表
        images = convert_from_path(pdf_file, dpi=dpi)

        # 遍历图片列表并保存
        for i, img in enumerate(images):
            # 格式化文件名，例如 '1.jpg', '2.jpg'
            output_name = f"{i + 1}.{output_type}"
            output_path = pjoin(output_dir, output_name)

            if output_type == 'png':
                img.save(output_path, 'PNG')
            elif output_type == 'jpg':
                img.save(output_path, 'JPEG')
            else:
                print(f"Warning: Unsupported output type '{output_type}'. Skipping.")

    # print(f"Successfully converted '{file}' to '{output_dir}'")


# --- 主程序逻辑 ---

# 定义你的路径

pptx_root_dir = "/data/ssd/Data/VideoAgent/generated_PPT_qwen-2.5-vl-7b"
output_base_dir = "/data/ssd/Data/VideoAgent/generated_frame_qwen-2.5-vl-7b"

# 遍历所有pptx文件
for pptx_file in os.listdir(pptx_root_dir):
    if pptx_file.endswith(".pptx"):
        pptx_path = pjoin(pptx_root_dir, pptx_file)

        # 获取文件名（不含扩展名），作为输出文件夹名
        file_name = Path(pptx_file).stem

        # 创建最终的输出路径，例如：/data/ssd/Data/VideoAgent/generated/test_ppt/frame
        output_frame_dir = pjoin(output_base_dir, file_name)
        try:
            # 调用转换函数
            ppt_to_multi_images(file=pptx_path, output_dir=output_frame_dir, dpi=150, output_type='jpg')
        except:
            print(output_frame_dir)
print("所有PPTX文件处理完毕！")