"""
🎬 Paper2Video - new_pipeline_multi.py
===========================================

这个文件是Paper2Video项目的核心pipeline，专门用于将学术论文转换为动态视频演示。
主要功能包括：
- 论文解析和内容提取
- 图片/表格智能过滤
- 多页PPT生成
- 学术旁白生成
- TTS语音合成
- 视频生成（支持Manim动画）

作者: Paper2Video Team
版本: 1.0
"""

# ==================== 导入模块 ====================
# 论文解析相关
from core.parse_raw import parse_raw, gen_image_and_table
# 大纲布局生成
from core.gen_outline_layout import filter_image_table, gen_outline_layout_v3
# 工具函数
from utils.wei_utils import get_agent_config, utils_functions, run_code, style_bullet_content, scale_to_target_area, \
    char_capacity
# 树形布局算法
from core.tree_split_layout import main_train, main_inference, get_arrangments_in_inches, split_textbox, \
    to_inches
# 自定义pipeline工具（包含TTS和旁白生成）
from core.my_pipeline_utils import run_filter_image_table, generate_narration_with_agent, generate_ppt_from_agent, synthesize_tts_with_openai, render_pptx_from_json, generate_bullet_points_with_agent
# PPT代码生成
from core.gen_pptx_code import generate_poster_code
# 多页PPT代码生成
from core.gen_pptx_code_multi import generate_multislide_ppt_code
# 视频处理工具
from utils.src.utils import ppt_to_images, ppt_to_multi_images, images_to_video
# 海报内容生成
# 删除bullet生成导入，改为直接使用raw_content
# 消融研究工具
from utils.ablation_utils import no_tree_get_layout

# 标准库导入
import subprocess
import os
import argparse
import json
import time
import glob

# ==================== 常量定义 ====================
units_per_inch = 25  # 每英寸的像素单位数

# ==================== 主题配置 ====================
# 主题配置将通过JSON文件加载，不再硬编码
from utils.theme_utils import load_theme_config, get_theme_colors, get_slide_dimensions, create_theme_dict
# 路径管理工具
from utils.path_utils import get_paper_name_from_path, create_output_dirs, get_file_path, save_json_file, load_json_file, file_exists

if __name__ == '__main__':
    # ==================== 命令行参数解析 ====================
    parser = argparse.ArgumentParser(description='🎬 Paper2Video - 视频生成Pipeline')
    
    # 基本参数
    parser.add_argument('--poster_path', type=str, 
                       help='论文PDF文件路径 (必需)')
    parser.add_argument('--model_name_t', type=str, default='4o',
                       help='文本模型名称 (默认: 4o)')
    parser.add_argument('--model_name_v', type=str, default='4o',
                       help='视觉模型名称 (默认: 4o)')
    parser.add_argument('--index', type=int, default=0,
                       help='输出文件索引 (默认: 0)')
    parser.add_argument('--poster_name', type=str, default=None,
                       help='自定义输出名称')
    parser.add_argument('--tmp_dir', type=str, default='tmp',
                       help='临时目录 (默认: tmp)')
    
    # 尺寸参数
    parser.add_argument('--poster_width_inches', type=int, default=None,
                       help='海报宽度 (英寸)')
    parser.add_argument('--poster_height_inches', type=int, default=None,
                       help='海报高度 (英寸)')
    
    # 调试参数
    parser.add_argument('--no_blank_detection', action='store_true',
                       help='禁用空白检测 (溢出严重时使用)')
    parser.add_argument('--ablation_no_tree_layout', action='store_true',
                       help='消融研究: 禁用树形布局')
    parser.add_argument('--ablation_no_commenter', action='store_true',
                       help='消融研究: 禁用评论器')
    parser.add_argument('--ablation_no_example', action='store_true',
                       help='消融研究: 禁用示例')
    
    # 视频参数
    parser.add_argument('--use_manim', action='store_true',
                       help='启用Manim动画')
    parser.add_argument('--generate', action='store_true',
                       help='启用视频生成 (必需)')
    parser.add_argument('--fps', type=int, default=1,
                       help='视频帧率 ')
    
    # 过滤参数
    parser.add_argument('--filter', action='store_true',
                       help='禁用图片和表格过滤，使用原始图片和表格')
    
    parser.add_argument('--use_bullet_points', action='store_true',
                       help='要点生成')

    args = parser.parse_args()
    start_time = time.time()

    # ==================== 初始化设置 ====================
    # 创建临时目录
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # 初始化日志记录
    detail_log = {}

    # 获取模型配置
    agent_config_t = get_agent_config(args.model_name_t)  # 文本模型配置
    agent_config_v = get_agent_config(args.model_name_v)  # 视觉模型配置
    
    # 处理海报名称
    poster_name = args.poster_path.split('/')[-2].replace(' ', '_')
    if args.poster_name is None:
        args.poster_name = poster_name
    else:
        poster_name = args.poster_name
    
    # ==================== 尺寸计算 ====================
    # 查找meta.json文件获取尺寸信息
    meta_json_path = args.poster_path.replace('paper.pdf', 'meta.json')
    
    # 确定海报尺寸（优先级：命令行参数 > meta.json > 默认值）
    if args.poster_width_inches is not None and args.poster_height_inches is not None:
        # 使用命令行指定的尺寸
        poster_width = args.poster_width_inches * units_per_inch
        poster_height = args.poster_height_inches * units_per_inch
    elif os.path.exists(meta_json_path):
        # 使用meta.json中的尺寸
        meta_json = json.load(open(meta_json_path, 'r'))
        poster_width = meta_json['width']
        poster_height = meta_json['height']
    else:
        # 使用默认尺寸 (48x36 英寸)
        poster_width = 48 * units_per_inch
        poster_height = 36 * units_per_inch

    # 缩放到目标区域
    poster_width, poster_height = scale_to_target_area(poster_width, poster_height)
    poster_width_inches = to_inches(poster_width, units_per_inch)
    poster_height_inches = to_inches(poster_height, units_per_inch)

    # 尺寸限制：如果超过56英寸，进行等比例缩放
    if poster_width_inches > 56 or poster_height_inches > 56:
        # 计算缩放因子（以较长边为准）
        if poster_width_inches >= poster_height_inches:
            scale_factor = 56 / poster_width_inches
        else:
            scale_factor = 56 / poster_height_inches

        # 应用缩放
        poster_width_inches *= scale_factor
        poster_height_inches *= scale_factor

        # 转换回内部单位
        poster_width = poster_width_inches * units_per_inch
        poster_height = poster_height_inches * units_per_inch

    print(f'📏 Poster size: {poster_width_inches} x {poster_height_inches} inches')

    # ==================== Token统计初始化 ====================
    total_input_tokens_t, total_output_tokens_t = 0, 0   # 文本模型token统计
    total_input_tokens_v, total_output_tokens_v = 0, 0   # 视觉模型token统计

    # ==================== 主要处理流程 ====================
    if args.generate:
        # 🔍 Step 1: 论文解析
        print('🔍 Step 1: Parse the raw poster')
        input_token, output_token, raw_result = parse_raw(args, agent_config_t, version=2)
        total_input_tokens_t += input_token
        total_output_tokens_t += output_token

        # 生成图片和表格
        _, _, images, tables = gen_image_and_table(args, raw_result)
        print(f'📊 Parsing token consumption: {input_token} -> {output_token}')

        detail_log['parser_in_t'] = input_token
        detail_log['parser_out_t'] = output_token

        # 🖼️ Step 2: 图片/表格过滤
        if args.filter:
            print('🖼️ Step 2: Filter unnecessary images and tables')
            input_token, output_token = filter_image_table(args, agent_config_t)
            total_input_tokens_t += input_token
            total_output_tokens_t += output_token
            print(f'🔍 Filter figures token consumption: {input_token} -> {output_token}')
            detail_log['filter_in_t'] = input_token
            detail_log['filter_out_t'] = output_token
        else:
            print('🖼️ Step 2: Skip image and table filtering (using original images and tables)')
            # 直接复制原始文件作为过滤后的文件
            import shutil
            import os
            
            # 获取论文名称
            paper_name = get_paper_name_from_path(args.poster_path)
            
            # 确保目录存在
            create_output_dirs(args.model_name_t, args.model_name_v, paper_name)
            
            # 复制原始图片文件
            if file_exists('images_and_tables', paper_name, f'{args.poster_name}_images.json', args.model_name_t, args.model_name_v):
                source_path = get_file_path('images_and_tables', paper_name, f'{args.poster_name}_images.json', args.model_name_t, args.model_name_v)
                target_path = get_file_path('images_and_tables', paper_name, f'{args.poster_name}_images_filtered.json', args.model_name_t, args.model_name_v)
                shutil.copy(source_path, target_path)
                print(f'✅ Copied original images to filtered images')
            
            # 复制原始表格文件
            if file_exists('images_and_tables', paper_name, f'{args.poster_name}_tables.json', args.model_name_t, args.model_name_v):
                source_path = get_file_path('images_and_tables', paper_name, f'{args.poster_name}_tables.json', args.model_name_t, args.model_name_v)
                target_path = get_file_path('images_and_tables', paper_name, f'{args.poster_name}_tables_filtered.json', args.model_name_t, args.model_name_v)
                shutil.copy(source_path, target_path)
                print(f'✅ Copied original tables to filtered tables')
            
            # 设置token消耗为0（因为没有调用LLM）
            input_token, output_token = 0, 0
            detail_log['filter_in_t'] = input_token
            detail_log['filter_out_t'] = output_token

        # 📝 Step 3: 生成大纲
        print('📝 Step 3: Generate outline')
        input_token, output_token, panels, figures = gen_outline_layout_v3(args, agent_config_t)
        print('-------------------------------------')
        print('📋 panels', panels)
        print('🖼️ figures', figures)

        total_input_tokens_t += input_token
        total_output_tokens_t += output_token
        print(f'📝 Outline token consumption: {input_token} -> {output_token}')

        detail_log['outline_in_t'] = input_token
        detail_log['outline_out_t'] = output_token
        # 🎨 Step 4: 布局学习与生成
        if args.ablation_no_tree_layout:
            # 消融研究：不使用树形布局
            print('🔬 Ablation: No tree layout')
            panel_arrangement, figure_arrangement, text_arrangement, input_token, output_token = no_tree_get_layout(
                poster_width,
                poster_height,
                panels,
                figures,
                agent_config_t
            )
            total_input_tokens_t += input_token
            total_output_tokens_t += output_token
            print(f'🔬 No tree layout token consumption: {input_token} -> {output_token}')
            detail_log['no_tree_layout_in_t'] = input_token
            detail_log['no_tree_layout_out_t'] = output_token
        else:
            # 正常流程：使用树形布局
            print('🎨 Step 4: Learn and generate layout')
            
            # 训练布局模型
            panel_model_params, figure_model_params = main_train()

            # 使用训练好的模型进行布局推理
            panel_arrangement, figure_arrangement, text_arrangement = main_inference(
                panels,
                panel_model_params,
                figure_model_params,
                poster_width,
                poster_height,
                shrink_margin=3  # 边距收缩
            )

            # 处理标题文本框：将其分为上下两部分
            text_arrangement_title = text_arrangement[0]  # 获取标题文本框
            text_arrangement = text_arrangement[1:]       # 移除标题，保留其他文本框
            # 将标题文本框按0.8比例分割为上下两部分
            text_arrangement_title_top, text_arrangement_title_bottom = split_textbox(
                text_arrangement_title,
                0.8
            )
            # 将分割后的标题文本框重新添加到列表开头
            text_arrangement = [text_arrangement_title_top, text_arrangement_title_bottom] + text_arrangement
        # ==================== 调试输出 ====================
        print('\n' + '='*50)
        print('📋 panel_arrangement', panel_arrangement)
        print('='*50)
        print('🖼️ figure_arrangement', figure_arrangement)
        print('='*50)
        print('📝 text_arrangement', text_arrangement)
        print('='*50)
        print(f'📊 Token统计: {input_token} -> {output_token}')
        print('='*50)


        # ==================== 图片路径处理 ====================
        # 为每个图片/表格安排添加实际文件路径
        for i in range(len(figure_arrangement)):
            panel_id = figure_arrangement[i]['slide_id']
            panel_section_name = panels[panel_id]['section_name']
            figure_info = figures[panel_section_name]
            
            # 处理图片
            if 'image' in figure_info:
                figure_id = figure_info['image']
                # 处理不同的ID格式（字符串或数字）
                if not figure_id in images:
                    figure_path = images[str(figure_id)]['image_path']
                else:
                    figure_path = images[figure_id]['image_path']
            # 处理表格
            elif 'table' in figure_info:
                figure_id = figure_info['table']
                # 处理不同的ID格式（字符串或数字）
                if not figure_id in tables:
                    figure_path = tables[str(figure_id)]['table_path']
                else:
                    figure_path = tables[figure_id]['table_path']

            # 将文件路径添加到图片安排中
            figure_arrangement[i]['figure_path'] = figure_path

        # ==================== 字符容量计算 ====================
        # 为每个文本框计算可容纳的字符数
        for text_arrangement_item in text_arrangement:
            num_chars = char_capacity(
                bbox=(text_arrangement_item['x'], text_arrangement_item['y'], 
                      text_arrangement_item['height'], text_arrangement_item['width'])
            )
            text_arrangement_item['num_chars'] = num_chars

        # ==================== 单位转换 ====================
        # 将所有布局安排从像素单位转换为英寸单位
        width_inch, height_inch, panel_arrangement_inches, figure_arrangement_inches, text_arrangement_inches = get_arrangments_in_inches(
            poster_width, poster_height, panel_arrangement, figure_arrangement, text_arrangement, 25
        )

        # ==================== 结果保存 ====================
        # 保存树形分割结果到文件
        tree_split_results = {
            'poster_width': poster_width,                    # 海报宽度（像素）
            'poster_height': poster_height,                  # 海报高度（像素）
            'poster_width_inches': width_inch,               # 海报宽度（英寸）
            'poster_height_inches': height_inch,             # 海报高度（英寸）
            'panels': panels,
            'panel_arrangement': panel_arrangement,
            'figure_arrangement': figure_arrangement,
            'text_arrangement': text_arrangement,
            'panel_arrangement_inches': panel_arrangement_inches,
            'figure_arrangement_inches': figure_arrangement_inches,
            'text_arrangement_inches': text_arrangement_inches,
        }
        # 保存tree_split结果到新的目录结构
        save_json_file(tree_split_results, 'tree_splits', paper_name, f'{args.poster_name}_tree_split_{args.index}.json', args.model_name_t, args.model_name_v)

        print('# Step 5: Load theme configuration')
        # 加载主题配置
        theme_config = load_theme_config()
        theme_title_text_color, theme_title_fill_color = get_theme_colors(theme_config)
        slide_width_inch, slide_height_inch = get_slide_dimensions(theme_config)
        theme = create_theme_dict(theme_config)
        
        print(f'Theme loaded: {slide_width_inch}x{slide_height_inch} inches')
        print(f'Title colors: text={theme_title_text_color}, fill={theme_title_fill_color}')

        # 从新的目录结构加载文件
        raw_result = load_json_file('contents', paper_name, f'{args.poster_name}_raw_content.json', args.model_name_t, args.model_name_v)
        raw_content = load_json_file('contents', paper_name, f'{args.poster_name}_raw_content.json', args.model_name_t, args.model_name_v)
        equations = load_json_file('equations', paper_name, 'equations_metadata.json', args.model_name_t, args.model_name_v)
        
        # 加载tree_split结果
        tree_split_results = load_json_file('tree_splits', paper_name, f'{args.poster_name}_tree_split_{args.index}.json', args.model_name_t, args.model_name_v)
        figure_arrangement_inches = tree_split_results["figure_arrangement_inches"]
    else:
        # 获取论文名称
        paper_name = get_paper_name_from_path(args.poster_path)
        
        # 从新的目录结构加载文件
        raw_result = load_json_file('contents', paper_name, f'{args.poster_name}_raw_content.json', args.model_name_t, args.model_name_v)
        tree_split_results = load_json_file('tree_splits', paper_name, f'{args.poster_name}_tree_split_{args.index}.json', args.model_name_t, args.model_name_v)
        # 直接使用raw_content，不再需要bullet_content
        raw_content = load_json_file('contents', paper_name, f'{args.poster_name}_raw_content.json', args.model_name_t, args.model_name_v)
        images = load_json_file('images_and_tables', paper_name, f'{args.poster_name}_images.json', args.model_name_t, args.model_name_v)
        tables = load_json_file('images_and_tables', paper_name, f'{args.poster_name}_tables.json', args.model_name_t, args.model_name_v)
        equations = load_json_file('equations', paper_name, 'equations_metadata.json', args.model_name_t, args.model_name_v)

        panel_arrangement_inches = tree_split_results["panel_arrangement_inches"]
        text_arrangement_inches = tree_split_results["text_arrangement_inches"]
        figure_arrangement_inches = tree_split_results["figure_arrangement_inches"]
        width_inch = tree_split_results["poster_width_inches"]
        height_inch = tree_split_results["poster_height_inches"]

        theme_config = load_theme_config()

    # 使用新的目录结构
    out_dir = get_file_path('contents', paper_name, '', args.model_name_t, args.model_name_v)
    tts_output_dir = get_file_path('tts', paper_name, '', args.model_name_t, args.model_name_v)
    raw_content_path = get_file_path('contents', paper_name, f'{args.poster_name}_raw_content.json', args.model_name_t, args.model_name_v)
    
    
    print('# Step 6.1: Generate narration')
    narration_json_path = get_file_path('contents', paper_name, f'{args.poster_name}_narration.json', args.model_name_t, args.model_name_v)
    if file_exists('contents', paper_name, f'{args.poster_name}_narration.json', args.model_name_t, args.model_name_v):
        print(f'[Skip] Narration exists: {narration_json_path}')
        narration_json = load_json_file('contents', paper_name, f'{args.poster_name}_narration.json', args.model_name_t, args.model_name_v)
        narration_in_token = narration_out_token = 0
    else:
        raw_content = load_json_file('contents', paper_name, f'{args.poster_name}_raw_content.json', args.model_name_t, args.model_name_v)
        narration_json, narration_in_token, narration_out_token = generate_narration_with_agent(raw_content, agent_config_t)
        total_input_tokens_t += narration_in_token
        total_output_tokens_t += narration_out_token
        print(f'Narration token consumption: {narration_in_token} -> {narration_out_token}')
        detail_log['narration_in_t'] = narration_in_token
        detail_log['narration_out_t'] = narration_out_token
        save_json_file(narration_json, 'contents', paper_name, f'{args.poster_name}_narration.json', args.model_name_t, args.model_name_v)

        # 支持仅生成旁白后提前退出（用于批处理脚本）
        if os.environ.get('P2V_NARRATION_ONLY') == '1':
            print('[Narration-Only] 已根据环境变量 P2V_NARRATION_ONLY=1 在旁白生成后提前结束。')
            sys.exit(0)



    print('# Step 6.2: TTS')
    tts_audio_log_path = get_file_path('tts', paper_name, 'tts_audio_files.json', args.model_name_t, args.model_name_v)
    if file_exists('tts', paper_name, 'tts_audio_files.json', args.model_name_t, args.model_name_v):
        print(f'[Skip] TTS audio exists: {tts_audio_log_path}')
        tts_audio_files = load_json_file('tts', paper_name, 'tts_audio_files.json', args.model_name_t, args.model_name_v)
    else:
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            print('[TTS] OPENAI_API_KEY not set, skip TTS synthesis.')
            tts_audio_files = {}
        else:
            tts_audio_files = synthesize_tts_with_openai(narration_json, tts_output_dir, openai_api_key, voice="alloy", language="zh")
        save_json_file(tts_audio_files, 'tts', paper_name, 'tts_audio_files.json', args.model_name_t, args.model_name_v)



    print('# Step 6.3: Apply theme styles')
    # 主题样式已在配置中定义，不再需要手动应用
    print('Theme styles will be applied during PPT generation')

    # 如果启用要点生成，则生成要点内容
    if args.use_bullet_points:
        print('# Step 6.4: Generate bullet points content')
        bullet_content_path = get_file_path('contents', paper_name, f'{args.poster_name}_bullet_content.json', args.model_name_t, args.model_name_v)
        if file_exists('contents', paper_name, f'{args.poster_name}_bullet_content.json', args.model_name_t, args.model_name_v):
            print(f'[Skip] Bullet points content exists: {bullet_content_path}')
            bullet_content = load_json_file('contents', paper_name, f'{args.poster_name}_bullet_content.json', args.model_name_t, args.model_name_v)
            bullet_in_token = bullet_out_token = 0
        else:
            # 为每个section生成要点
            bullet_content = {"sections": []}
            bullet_in_token = 0
            bullet_out_token = 0
            
            # 处理每个section
            for i, section in enumerate(raw_content["sections"]):
                print(f'Generating bullet points for section: {section["title"]}')
                try:
                    # 为当前section生成要点，传递section title信息
                    bullet_points_json, in_token, out_token = generate_bullet_points_with_agent(
                        section["content"], 
                        agent_config_t, 
                        prompt_path='utils/prompt_templates/bullet_point_generator.yaml',
                        section_title=section["title"]
                    )
                    bullet_in_token += in_token
                    bullet_out_token += out_token
                    
                    # 将要点转换为文本格式
                    bullet_text = "\n".join([f"• {point}" for point in bullet_points_json.get("bullet_points", [])])
                    
                    # 创建新的section结构
                    new_section = {
                        "title": section["title"],
                        "content": bullet_text,
                        "bullet_points": bullet_points_json.get("bullet_points", [])
                    }
                    
                    bullet_content["sections"].append(new_section)
                    print(f'Generated {len(bullet_points_json.get("bullet_points", []))} bullet points')
                except Exception as e:
                    print(f'Error generating bullet points for section {section["title"]}: {e}')
                    # 如果生成失败，使用原始内容
                    bullet_content["sections"].append(section)
            
            # 保存要点内容
            save_json_file(bullet_content, 'contents', paper_name, f'{args.poster_name}_bullet_content.json', args.model_name_t, args.model_name_v)
            print(f'Bullet points token consumption: {bullet_in_token} -> {bullet_out_token}')
            total_input_tokens_t += bullet_in_token
            total_output_tokens_t += bullet_out_token
            detail_log['bullet_in_t'] = bullet_in_token
            detail_log['bullet_out_t'] = bullet_out_token

    print('# Step 7: Generate the PowerPoint')
    # 根据是否启用要点生成来选择内容源
    content_for_ppt = bullet_content if args.use_bullet_points else raw_content
    poster_code, ppt_page_content = generate_multislide_ppt_code(
        content_for_ppt,  # 使用要点内容或原始内容
        raw_result,
        equations,
        figure_arrangement_inches,
        theme_config,  # 传递主题配置
        save_path=f'{args.tmp_dir}/poster_multipage.pptx'
    )
    output, err = run_code(poster_code)
    if err is not None:
        raise RuntimeError(f'Error in generating PowerPoint: {err}')


    print('# Step 8: Create a folder in the output directory')
    output_dir = get_file_path('generated_posters', paper_name, '', args.model_name_t, args.model_name_v)

    print('# Step 9: Move poster.pptx to the output directory')
    pptx_path = os.path.join(output_dir, f'{poster_name}_multipage.pptx')  # 修改文件名以作区分
    os.rename(f'{args.tmp_dir}/poster_multipage.pptx', pptx_path)
    print(f'Poster PowerPoint saved to {pptx_path}')
    #quit()

    print('# Step 10: Convert the PowerPoint to images')
    # ppt_to_images(pptx_path, output_dir)
    ppt_to_multi_images(file=pptx_path, output_dir=output_dir, output_type='png', dpi=150)
    print(f'Poster images saved to {output_dir}')


    # ==================== Manim动画生成 ====================
    if args.use_manim:
        print('🎬 Optional Step 11.0: Create Manim video')
        
        # === 新增：使用 Agent1 和 Agent2 自动生成 Manim 代码 ===
        from utils.manim_agent_generator import generate_manim_with_agents, create_default_manim_script_file
        
        script_path = f'./data/{args.poster_name}/animation.py'
        
        # 检查是否已有动画脚本，如果没有则使用 Agent 生成
        if not os.path.exists(script_path):
            print('🤖 Step 11.0.1: 使用 AI Agents 生成 Manim 动画代码...')
            
            try:
                # 调用 Agent1 和 Agent2 生成动画代码
                manim_code, agent_input_tokens, agent_output_tokens = generate_manim_with_agents(
                    args=args,
                    raw_content=raw_content,
                    agent_config=agent_config_t
                )
                
                # 更新 token 统计
                total_input_tokens_t += agent_input_tokens
                total_output_tokens_t += agent_output_tokens
                detail_log['manim_agent_in_t'] = agent_input_tokens
                detail_log['manim_agent_out_t'] = agent_output_tokens
                
                # 保存生成的代码
                os.makedirs(os.path.dirname(script_path), exist_ok=True)
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(manim_code)
                
                print(f'✅ Agent 生成的 Manim 代码已保存到: {script_path}')
                print(f'🔢 Agent Token 消耗: {agent_input_tokens} -> {agent_output_tokens}')
                
            except Exception as e:
                print(f'⚠️ Agent 生成失败: {e}')
                print('🔄 使用默认 Manim 模板...')
                # 使用默认模板作为降级方案
                create_default_manim_script_file(script_path, args.poster_name)
                detail_log['manim_agent_in_t'] = 0
                detail_log['manim_agent_out_t'] = 0
        else:
            print(f'✅ 发现已存在的动画脚本: {script_path}')
            detail_log['manim_agent_in_t'] = 0
            detail_log['manim_agent_out_t'] = 0
        
        # 设置Manim输出路径
        media_dir = f'./data/{args.poster_name}'  # Manim输出的根目录
        output_filename = 'implementation.mp4'     # 期望的最终文件名
        manim_video_paths = glob.glob(os.path.join(media_dir, 'videos', '*', '*', output_filename))

        # 检查Manim视频文件是否已存在
        if len(manim_video_paths) == 0:
            print(f"🎬 视频文件不存在，正在调用 Manim 生成")

            # ==================== Manim命令构建 ====================
            # 定义Manim脚本和场景的相关路径与名称
            script_path = f'./data/{args.poster_name}/animation.py'  # 动画脚本路径
            
            # 动态生成场景类名（与 Agent 生成的类名保持一致）
            from utils.manim_agent_generator import sanitize_class_name
            scene_name = f"{sanitize_class_name(args.poster_name)}Animation"

            # 构建Manim命令
            # -ql: 低质量（渲染快），-qh: 高质量
            # -o: 指定输出文件名
            # --media_dir: 指定输出目录
            command = [
                "manim",
                script_path,
                scene_name,
                "-ql",                    # 低质量模式（快速渲染）
                "--media_dir", media_dir, # 输出目录
                "-o", output_filename,    # 输出文件名
            ]

            print(f"🎬 执行命令: {' '.join(command)}")

            try:
                # 执行Manim命令并捕获输出
                result = subprocess.run(
                    command,
                    check=True,           # 如果命令失败则抛出异常
                    capture_output=True,   # 捕获标准输出和错误
                    text=True,            # 将输出解码为文本
                    encoding='utf-8'
                )
                
                # 查找生成的视频文件
                import glob
                manim_video_path = glob.glob(os.path.join(media_dir, 'videos', '*', '*', output_filename))[0]
                print("✅ Manim 视频渲染成功！保存至：", manim_video_path)
                
            except FileNotFoundError:
                print("\n❌ [错误] 'manim' 命令未找到。")
                print("请确保你已经正确安装了 Manim，并且 'manim' 命令在系统的 PATH 环境变量中。")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ [错误] Manim 渲染过程中出错:")
                print("--- Manim STDOUT ---")
                print(e.stdout)
                print("--- Manim STDERR ---")
                print(e.stderr)
                # 出错后，将路径设为None，避免后续代码使用一个不存在的文件
                manim_video_path = None

        else:
            # 如果视频文件已存在，直接使用
            manim_video_path = manim_video_paths[0]
            print(f"✅ Manim视频已存在: {manim_video_path}")
    else:
        # 如果未启用Manim，设置为None
        manim_video_path = None

    # ==================== 最终视频生成 ====================
    # 计算生成的图片页数
    num_pages = len(glob.glob(os.path.join(output_dir, '*.png')))
    print('🎬 Step 11: Create Video')
    
    # 设置GIF叠加路径（可选）
    gif_path = f'./data/{args.poster_name}/kq.gif'

    # 导入视频生成模块
    from utils.video_generate import create_presentation_video

    # 创建最终视频
    create_presentation_video(
        args=args,                           # 命令行参数
        image_dir=output_dir,                # 图片目录
        tts_audio_files=tts_audio_files,    # TTS音频文件
        page_to_section_map=ppt_page_content, # 页面到section的映射
        output_video_path=output_dir + 'video.mp4', # 输出视频路径
        overlay_gif_path=gif_path,           # GIF叠加路径
        manim_video_path=manim_video_path,  # Manim动画路径
        fps=args.fps                         # 帧率
    )



    # ==================== 性能统计和日志记录 ====================
    # 计算总耗时
    end_time = time.time()
    time_taken = end_time - start_time

    # 保存运行日志
    log_file = os.path.join(output_dir, 'log.json')
    with open(log_file, 'w') as f:
        log_data = {
            'input_tokens_t': total_input_tokens_t,   # 文本模型输入token数
            'output_tokens_t': total_output_tokens_t, # 文本模型输出token数
            'input_tokens_v': total_input_tokens_v,   # 视觉模型输入token数
            'output_tokens_v': total_output_tokens_v, # 视觉模型输出token数
            'time_taken': time_taken,
        }
        json.dump(log_data, f, indent=4)

    # 日志
    detail_log_file = os.path.join(output_dir, 'detail_log.json')
    with open(detail_log_file, 'w') as f:
        json.dump(detail_log, f, indent=4)
        print(f'\n✅ 处理完成！')
    print(f'📁 输出目录: {output_dir}')
    print(f'🎬 视频文件: {output_dir}video.mp4')
    print(f'📊 总耗时: {time_taken:.2f} 秒')
    print(f'💰 Token消耗: {total_input_tokens_t + total_input_tokens_v} -> {total_output_tokens_t + total_output_tokens_v}')