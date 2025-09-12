from dotenv import load_dotenv
from utils.src.utils import get_json_from_response
from utils.src.model_utils import parse_pdf
import json
import random

from camel.models import ModelFactory
from camel.agents import ChatAgent
from tenacity import retry, stop_after_attempt
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem, DocItemLabel

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from pathlib import Path
import fitz
from PIL import Image
import PIL

from marker.models import create_model_dict

from utils.wei_utils import *

from utils.pptx_utils import *
from utils.critic_utils import *
# 路径管理工具
from utils.path_utils import get_paper_name_from_path, create_output_dirs, get_file_path, save_json_file
import torch
from jinja2 import Template
import io
import os
import re
import argparse

load_dotenv()
IMAGE_RESOLUTION_SCALE = 5.0

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def crop_equation(args, raw_source, raw_doc, agent, text_content):
    # --- 2. 结合 PyMuPDF 提取公式图片 ---

    print("\nStep 1.1: 遍历公式并使用 PyMuPDF 截图...")

    # 获取论文名称并使用新的路径结构
    paper_name = get_paper_name_from_path(args.poster_path)
    output_dir = get_file_path('equations', paper_name, '', args.model_name_t, args.model_name_v)
    formulas_data = {}
    formula_count = 0

    pdf_document = fitz.open(raw_source)     # 打开同一个PDF文件

    template = Template(open("utils/prompts/gen_equation_summary.txt").read())

    # 遍历 docling 识别出的所有文本项
    for text_item in raw_doc.texts:
        # 检查是否为公式
        if text_item.label == DocItemLabel.FORMULA:
            formula_count += 1
            # (可选，但建议保留) 打印找到的公式内容，方便调试
            print(f"找到公式 {formula_count}，正在处理: {text_item.orig}")

            # 获取公式的位置信息 (大部分信息在 provenance 中)
            # 注意：一个公式可能跨越多个 provenance，这里我们取第一个
            if not text_item.prov:
                print(f"公式 {formula_count} 缺少位置信息，跳过。")
                continue

            provenance = text_item.prov[0]
            page_num_0_based = provenance.page_no - 1
            if page_num_0_based < 0:
                print(f"公式 {formula_count} 的页码无效 ({provenance.page_no})，跳过。")
                continue

            bbox = provenance.bbox
            page = pdf_document.load_page(page_num_0_based)

            # 2. 现在 `page` 变量已定义，可以安全地进行坐标转换
            x0 = bbox.l
            x1 = bbox.r
            y0 = page.rect.height - bbox.t
            y1 = page.rect.height - bbox.b

            # 3. 创建最终的、坐标正确的矩形对象
            rect = fitz.Rect(x0, y0, x1, y1)

            # 安全检查
            if rect.is_empty:
                print(f"公式 {formula_count} 的边界框无效或为空，跳过截图。")
                continue

            # 从页面中根据边界框获取像素图
            zoom = 2
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=rect)

            # 定义输出图片路径
            output_path = os.path.join(output_dir, f"formula_{formula_count}.png")
            # 保存为PNG文件
            pix.save(output_path)

            # --- 2. 【新增】提取上下文文本 ---
            # 定义一个在公式上下左右扩展的区域来寻找上下文
            context_margin_vertical = 100  # 上下各扩展 50 个像素单位
            context_margin_horizontal = 50  # 左右各扩展100个单位，确保捕获整行

            # 使用 `+` 操作符来扩展矩形，更简洁
            context_rect = rect + (
            -context_margin_horizontal, -context_margin_vertical, context_margin_horizontal, context_margin_vertical)

            # 确保上下文区域不会超出页面边界
            context_rect.intersect(page.rect)

            # 获取这个区域内的所有文本，并按阅读顺序排序
            context_text = page.get_text("text", clip=context_rect, sort=True)

            # 清理多余的换行和空格，使其成为一段连贯的文字
            context_text = re.sub(r'\s*\n\s*', ' ', context_text).strip()
            print(f"📝 提取到的上下文: \"{context_text}\"")
            agent.reset()
            prompt = template.render(
                markdown_document=text_content,
                formula_snippet=context_text
            )
            response = agent.step(prompt)
            response_json = get_json_from_response(response.msgs[0].content)
            rewritten_context = response_json.get("rewrite_content", context_text)
            section_title = response_json.get("section", "Methodology")
            print(f"🧠 AI 重写后的上下文: \"{rewritten_context}\"")

            # 使用 page.number 可以获取到 0-based 的页码，加 1 后就是我们习惯的页码
            print(f"已成功保存: {output_path} (来自页面 {page.number + 1})")
            formula_info = {
                "equ_path": output_path,
                "context": rewritten_context,  # 新增的上下文信息
                "section": section_title,
                "page": page.number + 1,
                "width": pix.width,  # 从 pixmap 对象直接获取宽度
                "height": pix.height  # 从 pixmap 对象直接获取高度
            }
            # 使用字符串格式的 formula_count 作为 JSON 中的 key
            formulas_data[str(formula_count)] = formula_info

    # 在循环结束后关闭PDF文档
    pdf_document.close()
    print("\n所有公式图片提取完成。")
    json_filename = "equations_metadata.json"
    json_output_path = os.path.join(output_dir, json_filename)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formulas_data, f, ensure_ascii=False, indent=4)

    print(f"\n已将所有公式元数据保存到: {json_output_path}")



# @retry(stop=stop_after_attempt(5))
def parse_raw(args, actor_config, version=1):
    raw_source = args.poster_path
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")

    raw_result = doc_converter.convert(raw_source)
    raw_doc = raw_result.document
    raw_markdown = raw_doc.export_to_markdown()
    text_content = markdown_clean_pattern.sub("", raw_markdown)

    if len(text_content) < 500:
        print('\nParsing with docling failed, using marker instead\n')
        parser_model = create_model_dict(device='cuda', dtype=torch.float16)
        text_content, rendered = parse_pdf(raw_source, model_lst=parser_model, save_file=False)

    if version == 1:
        template = Template(open("utils/prompts/gen_poster_raw_content.txt").read())
    elif version == 2:
        template = Template(open("utils/prompts/gen_poster_raw_content_v2.txt").read())
    elif version == 3:
        template = Template(open("utils/prompts/gen_poster_raw_content_v3.txt").read())

    if args.model_name_t.startswith('vllm_qwen'):
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    else:
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )

    actor_sys_msg = 'You are the author of the paper, and you will create a poster for the paper.'

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
        token_limit=actor_config.get('token_limit', None)
    )

    while True:
        prompt = template.render(
            markdown_document=text_content, 
        )
        actor_agent.reset()
        print('---------------------- this is ---------prompt',prompt)
        response = actor_agent.step(prompt)
        print('---------------------- finish step agent----------')
        input_token, output_token = account_token(response)

        content_json = get_json_from_response(response.msgs[0].content)

        if len(content_json) > 0:
            break
        print('Error: Empty response, retrying...')
        if args.model_name_t.startswith('vllm_qwen'):
            text_content = text_content[:80000]

    has_title = False

    for section in content_json['sections']:
        if type(section) != dict or not 'title' in section or not 'content' in section:
            print(f"Ouch! The response is invalid, the LLM is not following the format :(")
            print('Trying again...')
            raise
        if 'title' in section['title'].lower():
            has_title = True

    if not has_title:
        print('Ouch! The response is invalid, the LLM is not following the format :(')
        raise

    crop_equation(args, raw_source, raw_doc, actor_agent, text_content)

    # 获取论文名称并使用新的路径结构
    paper_name = get_paper_name_from_path(args.poster_path)
    save_json_file(content_json, 'contents', paper_name, f'{args.poster_name}_raw_content.json', args.model_name_t, args.model_name_v)
    return input_token, output_token, raw_result


def gen_image_and_table(args, conv_res):
    input_token, output_token = 0, 0
    raw_source = args.poster_path

    # 获取论文名称并使用新的路径结构
    paper_name = get_paper_name_from_path(args.poster_path)
    output_dir = Path(get_file_path('images_and_tables', paper_name, '', args.model_name_t, args.model_name_v))

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = args.poster_name

    # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"{doc_filename}-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    tables = {}

    table_index = 1
    for table in conv_res.document.tables:
        caption = table.caption_text(conv_res.document)
        if len(caption) > 0:
            table_img_path = get_file_path('images_and_tables', paper_name, f'{args.poster_name}-table-{table_index}.png', args.model_name_t, args.model_name_v)
            table_img = PIL.Image.open(table_img_path)
            tables[str(table_index)] = {
                'caption': caption,
                'table_path': table_img_path,
                'width': table_img.width,
                'height': table_img.height,
                'figure_size': table_img.width * table_img.height,
                'figure_aspect': table_img.width / table_img.height,
            }

        table_index += 1

    images = {}
    image_index = 1
    for image in conv_res.document.pictures:
        caption = image.caption_text(conv_res.document)
        if len(caption) > 0:
            image_img_path = get_file_path('images_and_tables', paper_name, f'{args.poster_name}-picture-{image_index}.png', args.model_name_t, args.model_name_v)
            image_img = PIL.Image.open(image_img_path)
            images[str(image_index)] = {
                'caption': caption,
                'image_path': image_img_path,
                'width': image_img.width,
                'height': image_img.height,
                'figure_size': image_img.width * image_img.height,
                'figure_aspect': image_img.width / image_img.height,
            }
        image_index += 1

    save_json_file(images, 'images_and_tables', paper_name, f'{args.poster_name}_images.json', args.model_name_t, args.model_name_v)
    save_json_file(tables, 'images_and_tables', paper_name, f'{args.poster_name}_tables.json', args.model_name_t, args.model_name_v)

    return input_token, output_token, images, tables

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--poster_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='4o')
    parser.add_argument('--poster_path', type=str, required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    agent_config = get_agent_config(args.model_name)

    if args.poster_name is None:
        args.poster_name = args.poster_path.split('/')[-1].replace('.pdf', '').replace(' ', '_')

    # Parse raw content
    input_token, output_token = parse_raw(args, agent_config)

    # Generate images and tables
    _, _ = gen_image_and_table(args)

    print(f'Token consumption: {input_token} -> {output_token}')
