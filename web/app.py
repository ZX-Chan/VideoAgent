
import gradio as gr
import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
# 删除了 time 和 threading 的导入，因为在此实现中不再需要

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入项目模块
from utils.wei_utils import get_agent_config
from utils.src.utils import get_json_from_response
from camel.models import ModelFactory
from camel.agents import ChatAgent
from utils.wei_utils import account_token

# 移除了全局变量 current_task，因为状态将由生成器函数直接管理

# 全局变量存储对话历史
chat_history = []

def create_simple_agent(prompt_template: str, model_name: str = "4o") -> ChatAgent:
    """
    创建简单的ChatAgent
    """
    agent_config = get_agent_config(model_name)
    model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
        url=agent_config.get('url', None)
    )
    
    agent = ChatAgent(
        system_message="你是一个专业的论文转视频pipeline参数解析器。",
        model=model,
        message_window_size=10,
    )
    
    return agent, prompt_template

def parse_user_requirements(user_input: str) -> dict:
    """
    使用GPT解析用户自然语言输入，转换为pipeline参数
    """
    try:
        # 创建解析用户需求的agent
        parse_prompt = """
你是一个专业的论文转视频pipeline参数解析器。用户会提供自然语言描述，你需要将其转换为具体的pipeline运行参数。

可用的参数选项：
- use_manim: 是否使用manim动画 (true/false)
- generate: 是否生成视频 (默认true，必需参数)
- fps: 视频帧率 (默认4)
- model_name_t: 文本模型名称 (默认4o)
- model_name_v: 视觉模型名称 (默认4o)
- filter: 是否禁用图片和表格过滤 (true/false)
- poster_width_inches: 海报宽度英寸 (可选)
- poster_height_inches: 海报高度英寸 (可选)
- no_blank_detection: 禁用空白检测 (true/false)
- ablation_no_tree_layout: 禁用树形布局 (true/false)
- ablation_no_commenter: 禁用评论器 (true/false)
- ablation_no_example: 禁用示例 (true/false)
- index: 输出文件索引 (默认0)
- poster_name: 自定义输出名称 (可选)
- tmp_dir: 临时目录 (默认tmp)

用户输入: {user_input}

请以JSON格式返回解析结果，格式如下：
{{
    "use_manim": true/false,
    "generate": true/false,
    "fps": 4,
    "model_name_t": "4o",
    "model_name_v": "4o",
    "filter": false,
    "poster_width_inches": null,
    "poster_height_inches": null,
    "no_blank_detection": false,
    "ablation_no_tree_layout": false,
    "ablation_no_commenter": false,
    "ablation_no_example": false,
    "index": 0,
    "poster_name": null,
    "tmp_dir": "tmp",
    "additional_params": {{}}
}}

只返回JSON，不要其他内容。
"""
        
        agent, prompt = create_simple_agent(parse_prompt, "4o")
        prompt = prompt.format(user_input=user_input)
        
        agent.reset()
        response = agent.step(prompt)
        result = response.msgs[0].content
        
        # 尝试解析JSON结果
        try:
            # 提取JSON部分
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = result[json_start:json_end]
                params = json.loads(json_str)
                return params
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError:
            # 如果解析失败，返回默认参数
            return {
                "use_manim": False, "generate": True, "fps": 4,
                "model_name_t": "4o", "model_name_v": "4o", "filter": False,
                "poster_width_inches": None, "poster_height_inches": None,
                "no_blank_detection": False, "ablation_no_tree_layout": False,
                "ablation_no_commenter": False, "ablation_no_example": False,
                "index": 0, "poster_name": None, "tmp_dir": "tmp",
                "additional_params": {}
            }
            
    except Exception as e:
        print(f"解析用户需求时出错: {e}")
        return {
            "use_manim": False, "generate": True, "fps": 4,
            "model_name_t": "4o", "model_name_v": "4o", "filter": False,
            "poster_width_inches": None, "poster_height_inches": None,
            "no_blank_detection": False, "ablation_no_tree_layout": False,
            "ablation_no_commenter": False, "ablation_no_example": False,
            "index": 0, "poster_name": None, "tmp_dir": "tmp",
            "additional_params": {}
        }

def chat_with_gpt(user_input: str) -> str:
    """
    与GPT对话，维护对话历史并生成自然语言的需求总结
    """
    global chat_history
    
    try:
        # 创建需求确认的agent
        chat_prompt = """
你是一个友好的论文转视频需求分析师。用户会告诉你他们的需求，你需要：

1. 理解用户的需求
2. 用自然、友好的语言简要总结用户的需求
3. 如果需求不够明确，可以询问一些细节
4. 最后用自然语言总结一下用户要什么

⚠️ 重要提醒：
- 你的任务是确认和理解用户需求，不是提供技术教程
- 不要回复安装说明、代码示例、使用方法等技术内容
- 不要解释什么是Manim、如何安装、如何使用等
- 只关注用户想要什么效果，然后确认需求

请用自然、日常的语言回复，就像朋友之间的对话一样。例如：
- 用户说"使用manim动画，视频fps=1"，你应该回复：
  "好的，我理解你想要用manim制作动画，并且视频帧率设为1fps，这样会生成很慢的动画效果。还有其他要求吗？比如视频质量、模型选择等？"

- 用户说"生成简洁的演示视频"，你应该回复：
  "好的，你要生成简洁的演示视频。需要什么特殊效果吗？比如动画、特定的帧率？"

记住：你的回复应该是对话式的，不是教程式的。专注于理解用户需求，用自然语言确认和总结。
"""
        
        agent, prompt = create_simple_agent(chat_prompt, "4o")
        
        # 添加用户输入到对话历史
        chat_history.append(f"用户: {user_input}")
        
        # 构建完整的对话上下文，包含系统提示和用户输入
        full_context = prompt + "\n\n" + "\n".join(chat_history) + "\n\n助手: "
        
        agent.reset()
        response = agent.step(full_context)
        bot_response = response.msgs[0].content
        
        # 添加助手回复到对话历史
        chat_history.append(f"助手: {bot_response}")
        
        return bot_response
        
    except Exception as e:
        return f"抱歉，我在处理您的需求时遇到了问题: {str(e)}"

def run_pipeline(pdf_file, final_requirements: str, progress=gr.Progress(track_tqdm=True)):
    """
    运行pipeline的主要函数 (已修改为生成器)
    """
    # 初始状态更新
    yield None, "...", "...", progress(0, desc="开始处理...")
    
    if pdf_file is None:
        yield None, "请先上传PDF文件", "❌ 文件上传失败", progress(0, desc="文件上传失败")
        return

    if not os.path.exists(pdf_file.name):
        yield None, "PDF文件不存在", "❌ 文件上传失败", progress(0, desc="文件不存在")
        return
    
    file_size = os.path.getsize(pdf_file.name)
    if file_size > 15 * 1024 * 1024:  # 15MB
        yield None, "PDF文件过大 ( > 15MB )", "❌ 文件过大", progress(0, desc="文件过大")
        return

    if not final_requirements.strip():
        final_requirements = "生成简洁的演示视频"
    
    temp_dir = "" # 初始化变量
    try:
        yield None, "正在解析用户需求...", "等待中...", progress(0.1, desc="解析用户需求...")
        params = parse_user_requirements(final_requirements)
        
        temp_dir = tempfile.mkdtemp(prefix="paper2video_")
        pdf_path = os.path.join(temp_dir, "paper.pdf")
        shutil.copy2(pdf_file.name, pdf_path)
        
        yield None, "正在初始化pipeline...", "等待中...", progress(0.2, desc="初始化pipeline...")
        
        # 生成唯一的poster_name，避免文件冲突
        import time
        timestamp = int(time.time())
        unique_poster_name = f"paper_{timestamp}"
        
        cmd = [
            sys.executable, "-m", "PosterAgent.new_pipeline_multi",
            "--poster_path", pdf_path,
            "--poster_name", unique_poster_name,
            "--generate"
        ]
        
        if params.get("use_manim", False): cmd.append("--use_manim")
        if "fps" in params: cmd.extend(["--fps", str(params["fps"])])
        if "model_name_t" in params: cmd.extend(["--model_name_t", params["model_name_t"]])
        if "model_name_v" in params: cmd.extend(["--model_name_v", params["model_name_v"]])
        if params.get("filter", False): cmd.append("--filter")
        if params.get("poster_width_inches"): cmd.extend(["--poster_width_inches", str(params["poster_width_inches"])])
        if params.get("poster_height_inches"): cmd.extend(["--poster_height_inches", str(params["poster_height_inches"])])
        if params.get("no_blank_detection", False): cmd.append("--no_blank_detection")
        if params.get("ablation_no_tree_layout", False): cmd.append("--ablation_no_tree_layout")
        if params.get("ablation_no_commenter", False): cmd.append("--ablation_no_commenter")
        if params.get("ablation_no_example", False): cmd.append("--ablation_no_example")
        
        yield None, "正在运行pipeline...", "执行中...", progress(0.4, desc="运行pipeline (此步耗时较长)...")
        
        print(f"\n🚀 开始执行pipeline命令: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时读取日志，并缓慢推进进度条
        progress_amount = 0.4
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                # 在pipeline运行时，缓慢增加进度条，提供视觉反馈
                if progress_amount < 0.9:
                    progress_amount += 0.005 # 调整这个值可以改变进度条前进速度
                    yield None, "正在运行pipeline...", "执行中...", progress(progress_amount, desc="运行pipeline (此步耗时较长)...")

        stdout, _ = process.communicate()
        if process.returncode != 0:
            error_message = f"Pipeline运行失败。日志: {stdout}"
            yield None, error_message, "❌ Pipeline运行失败", progress(0, desc="Pipeline运行失败")
            return

        yield None, "正在查找生成的视频...", "即将完成...", progress(0.95, desc="查找视频...")
        
        video_path = None
        
        # 查找新生成的视频文件
        # pipeline使用get_file_path生成输出目录，格式类似 <4o_4o>_generated_posters/paper/
        video_path = None
        
        # 等待一下，确保文件系统更新
        time.sleep(2)
        
        # 查找生成的视频文件，使用pipeline的实际输出路径
        possible_video_paths = [
            os.path.join(project_root, f"<{params.get('model_name_t', '4o')}_{params.get('model_name_v', '4o')}>_generated_posters", unique_poster_name, "video.mp4"),  # pipeline实际输出路径
            os.path.join(project_root, f"<{params.get('model_name_t', '4o')}_{params.get('model_name_v', '4o')}>_generated_posters", unique_poster_name, f"{unique_poster_name}_multipage.mp4"),  # 可能的多页输出
            os.path.join(project_root, "web", "temp_video.mp4")  # 备用路径
        ]
        
        print(f"🔍 查找视频文件，检查路径:")
        for path in possible_video_paths:
            print(f"  - {path}")
        
        for video_file in possible_video_paths:
            if os.path.exists(video_file):
                file_size = os.path.getsize(video_file)
                if file_size > 1000:  # 文件大小大于1KB
                    video_path = video_file
                    print(f"✅ 找到视频文件: {video_file}, 大小: {file_size} bytes")
                    break
                else:
                    print(f"⚠️ 视频文件太小，可能生成失败: {video_file}, 大小: {file_size} bytes")
        
        if not video_path:
            print(f"❌ 未找到有效的视频文件")
            print(f"检查的路径: {possible_video_paths}")

        if video_path and os.path.exists(video_path):
            yield video_path, "视频生成完成！", "✅ 视频生成完成！", progress(1, desc="完成!")
        else:
            yield None, "未找到生成的视频文件", "❌ 未找到视频文件", progress(0, desc="未找到视频文件")
             
    except Exception as e:
        error_msg = f"处理过程中出错: {str(e)}"
        yield None, error_msg, f"❌ 处理出错", progress(0, desc="处理出错")
        
    finally:
        # 清理临时文件和旧输出目录
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"🗑️ 已清理临时目录: {temp_dir}")
            
            # 清理旧的输出目录，避免积累太多文件
            # 清理 <model>_generated_posters 目录下的旧文件
            current_time = time.time()
            for item in os.listdir(project_root):
                if item.endswith("_generated_posters") and os.path.isdir(os.path.join(project_root, item)):
                    generated_dir = os.path.join(project_root, item)
                    for subdir in os.listdir(generated_dir):
                        subdir_path = os.path.join(generated_dir, subdir)
                        if os.path.isdir(subdir_path) and subdir.startswith("paper_"):
                            # 检查目录是否超过1小时
                            dir_time = os.path.getctime(subdir_path)
                            if current_time - dir_time > 3600:  # 1小时 = 3600秒
                                try:
                                    shutil.rmtree(subdir_path)
                                    print(f"🗑️ 已清理旧输出目录: {subdir_path}")
                                except Exception as e:
                                    print(f"清理旧输出目录时出错: {e}")
        except Exception as cleanup_error:
            print(f"清理文件时出错: {cleanup_error}")

def create_interface():
    """
    创建Gradio界面 (已修复版本兼容性问题)
    """
    # CSS样式保持不变
    css = """
    .gradio-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .main-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    .title {
        text-align: center;
        color: #1a1a1a;
        font-size: 2.8em;
        font-weight: bold;
        margin-bottom: 30px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #4a4a4a;
        font-size: 1.3em;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    .upload-area {
        border: 3px dashed #4a90e2;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        background: linear-gradient(45deg, #ffffff, #f8f9fa);
        transition: all 0.3s ease;
        margin: 20px 0;
    }
    
    .upload-area:hover {
        border-color: #357abd;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        transform: translateY(-2px);
    }
    
    .input-area {
        background: #ffffff;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid #4a90e2;
        border: 1px solid #e1e5e9;
    }
    
    .video-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #e1e5e9;
    }
    
    .pdf-preview {
        background: #ffffff;
        border-radius: 15px;
        padding: 0;
        margin: 0;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        width: 100%;
        height: 200px;
        overflow: visible;
    }
    
    /* 新的PDF预览容器样式 */
    .pdf-preview-container {
        width: 100%;
        margin-top: 0;
        margin-bottom: 20px;
        overflow: visible;
        min-height: 200px;
    }
    
    /* 移除HTML组件的默认滚动条 */
    .gradio-container .pdf-preview {
        overflow: hidden !important;
        scrollbar-width: none !important; /* Firefox */
        -ms-overflow-style: none !important; /* IE and Edge */
    }
    
    .gradio-container .pdf-preview::-webkit-scrollbar {
        display: none !important; /* Chrome, Safari, Opera */
    }
    
    /* 修复对齐问题，确保上传区域和视频预览区域对齐 */
    .gradio-container .upload-area {
        margin-top: 0;
        margin-bottom: 20px;
    }
    
    /* 确保PDF预览区域和视频预览区域高度一致 */
    .pdf-preview {
        margin-top: 0;
        margin-bottom: 20px;
    }
    
    /* 隐藏PDF输入组件的默认样式 */
    .gradio-container .upload-area input[type="file"] {
        opacity: 0;
        position: absolute;
        z-index: 1;
        width: 100%;
        height: 100%;
        cursor: pointer;
    }
    
    .gradio-container .upload-area label {
        display: none !important;
    }
    
    /* 美化PDF输入组件 */
    .gradio-container .upload-area {
        position: relative;
        min-height: 80px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    

    

    
    .progress-bar {
        background: linear-gradient(90deg, #4a90e2, #357abd);
        border-radius: 10px;
        height: 8px;
        margin: 10px 0;
    }
    
    .button-primary {
        background: linear-gradient(45deg, #4a90e2, #357abd);
        border: none;
        border-radius: 25px;
        color: white;
        padding: 15px 30px;
        font-size: 1.1em;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
    }
    
    .button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
    }
    
    .tips-container {
        background: rgba(74, 144, 226, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border-left: 4px solid #4a90e2;
        border: 1px solid #e1e5e9;
    }
    
    .tips-container h4 {
        color: #1a1a1a;
        margin-bottom: 15px;
        font-weight: bold;
    }
    
    .tips-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .tips-list li {
        padding: 8px 0;
        color: #4a4a4a;
        position: relative;
        padding-left: 20px;
    }
    
    .tips-list li:before {
        content: "💡";
        position: absolute;
        left: 0;
        color: #4a90e2;
    }
    
    /* 确保所有文字都有足够的对比度 */
    .gradio-container label {
        color: #1a1a1a !important;
        font-weight: 500;
    }
    
    .gradio-container input, .gradio-container textarea {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        border: 1px solid #e1e5e9 !important;
    }
    
    .gradio-container input:focus, .gradio-container textarea:focus {
        border-color: #4a90e2 !important;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2) !important;
    }
    
    /* 隐藏PDF上传和视频预览区域的小黑框 */
    .gradio-container .upload-area label,
    .gradio-container .video-container label {
        display: none !important;
    }
    
    /* 为视频区域添加默认提示内容 */
    .video-container:has(video:not([src]))::before {
        content: "🎬 视频将在这里显示";
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: #4a4a4a;
        font-size: 1.2em;
        font-weight: 500;
        text-align: center;
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 20px;
        border: 2px dashed #4a90e2;
    }
    """

    with gr.Blocks(css=css, title="Paper2Video - 论文转视频工具") as interface:
        gr.HTML("""
        <div class="title">📄 Paper2Video</div>
        <div class="subtitle">将学术论文转换为精美的演示视频</div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<h3 style="color: #2c3e50; margin-bottom: 20px;">📤 上传论文</h3>')
                # PDF上传组件
                pdf_input = gr.File(
                    label="拖拽PDF文件到这里",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_classes=["upload-area"]
                )
                
                # PDF预览区域
                pdf_preview = gr.HTML(
                    value="",
                    visible=False,
                    elem_classes=["pdf-preview"]
                )
                
                gr.HTML('<h3 style="color: #2c3e50; margin-bottom: 20px;">💬 需求确认</h3>')
                
                # 用户输入框
                user_input = gr.Textbox(
                    label="请输入您的需求",
                    placeholder="例如：请使用manim动画，生成4fps的视频...",
                    lines=3
                )
                
                # 按钮
                with gr.Row():
                    chat_btn = gr.Button("💬 发送消息")
                
                # 最终需求
                final_requirements = gr.Textbox(
                    label="最终需求",
                    value="等待确认...",
                    interactive=False,
                    lines=3
                )
                
                # 确认需求按钮
                confirm_btn = gr.Button("✅ 确认需求并开始处理")
                
                # 【已修复】: 移除了不支持的 'label' 参数
                #gr.Markdown("#### 处理进度") # 使用Markdown添加一个标题
                progress_bar = gr.Progress()

            with gr.Column(scale=1):
                gr.HTML('<h3 style="color: #2c3e50; margin-bottom: 20px;">🎬 视频预览</h3>')
                video_output = gr.Video(
                    label="生成的视频",
                    height=400,
                    elem_classes=["video-container"]
                )
                
                status_output = gr.Textbox(
                    label="处理状态",
                    value="等待开始处理...",
                    interactive=False
                )
                result_info = gr.Textbox(
                    label="处理结果",
                    value="等待处理...",
                    interactive=False
                )
        
        # 事件处理逻辑
        # 发送聊天消息
        chat_btn.click(
            fn=chat_with_gpt,
            inputs=[user_input],
            outputs=[final_requirements],
            show_progress=False
        )
        
        # 确认需求并开始处理
        confirm_btn.click(
            fn=run_pipeline,
            inputs=[pdf_input, final_requirements],
            outputs=[video_output, status_output, result_info],
            show_progress="hidden"
        )
        
        # 添加PDF上传后的内容预览功能
        def show_pdf_preview(file):
            if file is not None:
                try:
                    # 获取文件信息
                    file_name = os.path.basename(file.name) if hasattr(file, 'name') else "未知文件"
                    file_size = os.path.getsize(file.name) if hasattr(file, 'name') else 0
                    
                    # 直接显示文件信息，不尝试PDF预览
                    info_html = f"""
                    <div class="pdf-preview-container" style="overflow: visible;">
                        <div style="width: 100%; height: 200px; border: 1px solid #e1e5e9; border-radius: 15px; display: flex; align-items: center; justify-content: center; background: #ffffff; position: relative;">
                            <div style="text-align: center; color: #6c757d; transform: translateY(-15px);">
                                <h4 style="color: #1a1a1a; font-weight: bold; margin-bottom: 15px;">📄 PDF文件已上传</h4>
                                <p style="font-size: 1.2em; color: #2c3e50; margin: 8px 0;">{file_name}</p>
                                <p style="font-size: 1em; color: #34495e; margin: 6px 0;">文件大小: {file_size / 1024 / 1024:.2f} MB</p>
                                <div style="margin-top: 8px; padding: 8px; background: #e3f2fd; border-radius: 10px; border-left: 4px solid #4a90e2;">
                                    <p style="font-size: 0.9em; color: #1565c0; margin: 0;">
                                        PDF文件已成功上传，您现在可以在下方输入需求并开始处理
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                    return gr.File(visible=False), gr.HTML(value=info_html, visible=True)
                        
                except Exception as e:
                    # 如果出现其他错误，显示基本文件信息
                    error_html = f"""
                    <div class="pdf-preview-container" style="overflow: visible;">
                        <div style="width: 100%; height: 200px; border: 1px solid #e1e5e9; border-radius: 15px; display: flex; align-items: center; justify-content: center; background: #ffffff; position: relative;">
                            <div style="text-align: center; color: #6c757d;">
                                <h4>📄 PDF上传状态</h4>
                                <p>✅ 文件已上传</p>
                                <p style="font-size: 0.9em; color: #adb5bd;">预览功能暂时不可用</p>
                                <p style="font-size: 0.8em; color: #95a5a6;">错误: {str(e)}</p>
                            </div>
                        </div>
                    </div>
                    """
                    return gr.File(visible=False), gr.HTML(value=error_html, visible=True)
            else:
                # 如果没有文件，显示默认的上传界面
                return gr.File(visible=True), gr.HTML(visible=False)
        
        # 处理PDF文件变化
        pdf_input.change(
            fn=show_pdf_preview,
            inputs=[pdf_input],
            outputs=[pdf_input, pdf_preview]
        )
        
        # 使用提示部分保持不变
        gr.HTML("""
        <div class="tips-container">
            <h4>💡 使用提示</h4>
            <p style="color: #7f8c8d; margin-bottom: 15px;">您可以用自然语言描述您的需求，比如：</p>
            <ul class="tips-list">
                <li>"使用manim动画，生成高清视频"</li>
                <li>"帧率设为4fps，使用4o模型"</li>
                <li>"生成简洁的演示视频"</li>
                <li>"添加字幕，生成适合演示的视频"</li>
                <li>"禁用图片过滤，使用原始图片"</li>
                <li>"禁用空白检测，处理复杂布局"</li>
                <li>"使用树形布局，优化内容结构"</li>
            </ul>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )