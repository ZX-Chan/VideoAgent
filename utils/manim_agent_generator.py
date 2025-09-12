
import os
import json
import tempfile
from typing import Tuple, Dict, Any, Optional
from camel.models import ModelFactory
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from utils.wei_utils import account_token
from utils.src.utils import get_json_from_response


def generate_manim_with_agents(
    args,
    raw_content: Dict[str, Any],
    agent_config: Dict[str, Any]
) -> Tuple[str, int, int]:
    """
    使用 Agent1 和 Agent2 自动生成完整的 Manim 动画代码
    
    Args:
        args: 命令行参数对象
        raw_content: 论文的原始内容数据
        agent_config: Agent 模型配置
        
    Returns:
        (manim_code_string, total_input_tokens, total_output_tokens)
    """
    print("🤖 开始调用 AI Agents 生成 Manim 动画代码...")
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    try:
        # Step 1: 调用 Agent1 进行动画规划
        print("📋 Step 1: 调用 Agent1 (动画规划师) 分析论文并生成动画规划...")
        planning_json, tokens1_in, tokens1_out = call_agent1_animation_planner(
            raw_content, args, agent_config
        )
        total_input_tokens += tokens1_in
        total_output_tokens += tokens1_out
        print(f"✅ Agent1 完成，生成了 {len(planning_json.get('scene_sequence', []))} 个动画场景")
        
        # Step 2: 调用 Agent2 生成 Manim 代码
        print("💻 Step 2: 调用 Agent2 (代码生成器) 根据规划生成 Manim 代码...")
        manim_code, tokens2_in, tokens2_out = call_agent2_code_generator(
            planning_json, args, agent_config
        )
        total_input_tokens += tokens2_in
        total_output_tokens += tokens2_out
        print("✅ Agent2 完成，生成了完整的 Manim 动画代码")
        
        # Step 3: 代码验证
        if validate_manim_code(manim_code):
            print("✅ 生成的 Manim 代码通过基础验证")
            return manim_code, total_input_tokens, total_output_tokens
        else:
            raise ValueError("生成的代码未通过验证")
            
    except Exception as e:
        print(f"❌ Agent 生成失败: {e}")
        print("🔄 使用默认模板作为降级方案...")
        default_code = create_default_manim_script(args.poster_name)
        return default_code, total_input_tokens, total_output_tokens


def call_agent1_animation_planner(
    raw_content: Dict[str, Any],
    args,
    agent_config: Dict[str, Any]
) -> Tuple[Dict[str, Any], int, int]:
    """
    调用 Agent1 进行动画规划
    
    Args:
        raw_content: 论文原文内容
        args: 包含poster_name等参数
        agent_config: Agent配置
        
    Returns:
        (planning_json, input_tokens, output_tokens)
    """
    # 加载 Agent1 提示词
    agent1_prompt_path = 'utils/prompts/agent1.txt'
    with open(agent1_prompt_path, 'r', encoding='utf-8') as f:
        agent1_system_prompt = f.read()
    
    # 创建 Agent1 模型
    model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
        url=agent_config.get('url', None)
    )
    
    agent1 = ChatAgent(
        system_message=agent1_system_prompt,
        model=model,
        message_window_size=None,
    )
    
    # 提取论文原文内容
    paper_original_text = extract_paper_original_text(raw_content)
    
    # 获取目标图表信息（用于生成简短需求）
    target_figure_info = select_target_figure(raw_content, args)
    
    # 构建简短需求
    short_requirement = generate_short_requirement(target_figure_info, args.poster_name)
    
    # 按照新要求构建用户输入：论文原文 + 简短需求
    user_input = f"""
# 论文原文
{paper_original_text}

# 动画需求
{short_requirement}

请基于以上论文原文内容和动画需求，生成详细的动画规划JSON。
"""
    
    # 调用 Agent1
    response = agent1.step(user_input)
    input_tokens, output_tokens = account_token(response)
    
    # 解析响应为 JSON
    planning_json = get_json_from_response(response.msgs[0].content)
    
    if not planning_json:
        # 如果JSON解析失败，创建一个基础的规划结构
        planning_json = create_fallback_planning(args.poster_name)
    
    return planning_json, input_tokens, output_tokens


def call_agent2_code_generator(
    planning_json: Dict[str, Any],
    args,
    agent_config: Dict[str, Any]
) -> Tuple[str, int, int]:
    """
    调用 Agent2 生成 Manim 代码
    
    Args:
        planning_json: Agent1产生的JSON规划
        args: 包含poster_name等参数
        agent_config: Agent配置
        
    Returns:
        (manim_code, input_tokens, output_tokens)
    """
    # 加载 Agent2 提示词
    agent2_prompt_path = 'utils/prompts/agent2.txt'
    with open(agent2_prompt_path, 'r', encoding='utf-8') as f:
        agent2_system_prompt = f.read()
    
    # 创建 Agent2 模型
    model = ModelFactory.create(
        model_platform=agent_config['model_platform'],
        model_type=agent_config['model_type'],
        model_config_dict=agent_config['model_config'],
        url=agent_config.get('url', None)
    )
    
    agent2 = ChatAgent(
        system_message=agent2_system_prompt,
        model=model,
        message_window_size=None,
    )
    
    # 获取相应的figure图片信息
    figure_image_info = get_target_figure_image(args)
    
    # 构建用户输入：Agent1产生的JSON + 相应figure图片
    scene_class_name = f"{sanitize_class_name(args.poster_name)}Animation"
    
    user_input = f"""
# Agent1产生的动画规划JSON
{json.dumps(planning_json, indent=2, ensure_ascii=False)}

# 相应figure图片信息
{figure_image_info}

# 代码生成要求
- 场景类名: {scene_class_name}
- 继承自 Scene 类
- 包含完整的 construct 方法
- 使用标准 Manim 语法
- 代码必须可以直接执行
- 参考图片信息来准确还原视觉效果

# Manim 标准形状类（只使用这些）
- Rectangle: 矩形
- Circle: 圆形
- Square: 正方形
- RegularPolygon: 正多边形（可用于三角形、六角形等）
- Ellipse: 椭圆
- Line: 直线
- Arrow: 箭头
- Text: 文本
- Dot: 点

# 注意：禁止使用 Diamond, Triangle, Hexagon 等不存在的类

请根据JSON规划和图片信息生成完整的Python代码文件内容。
"""
    
    # 调用 Agent2
    response = agent2.step(user_input)
    input_tokens, output_tokens = account_token(response)
    
    # 提取代码内容
    manim_code = extract_python_code(response.msgs[0].content)
    
    if not manim_code or len(manim_code) < 100:
        # 如果代码提取失败或太短，使用默认模板
        manim_code = create_default_manim_script(args.poster_name)
    
    return manim_code, input_tokens, output_tokens


def extract_paper_original_text(raw_content: Dict[str, Any]) -> str:
    """提取论文原文内容"""
    try:
        sections = raw_content.get('sections', [])
        
        # 简洁地提取所有章节的原文内容
        original_text = ""
        
        for section in sections:
            title = section.get('title', '')
            content = section.get('content', '')
            
            original_text += f"\n## {title}\n"
            original_text += f"{content}\n"
        
        return original_text.strip()
        
    except Exception as e:
        return f"论文原文提取错误: {e}"


def generate_short_requirement(target_figure_info: str, poster_name: str) -> str:
    """根据目标图表信息生成简短的动画需求"""
    try:
        # 从图表信息中提取关键内容来生成需求
        if "figure" in target_figure_info.lower() or "图" in target_figure_info:
            # 如果有具体的图表信息
            if "architecture" in target_figure_info.lower() or "框架" in target_figure_info:
                requirement = f"根据论文中的系统架构图，给出详细的动态演示，要讲清楚整个系统的工作流程和各组件的交互关系"
            elif "flow" in target_figure_info.lower() or "流程" in target_figure_info:
                requirement = f"根据论文中的算法流程图，给出详细的动态演示，要讲清楚算法的执行步骤和数据流向"
            elif "model" in target_figure_info.lower() or "模型" in target_figure_info:
                requirement = f"根据论文中的模型结构图，给出详细的动态演示，要讲清楚模型的组成和工作原理"
            else:
                requirement = f"根据论文中的核心图表，给出详细的动态演示，要讲清楚图表所表达的核心思想和技术方法"
        else:
            # 如果没有具体图表信息，生成通用需求
            requirement = f"根据论文《{poster_name}》的核心内容，创建一个动态演示动画，要讲清楚论文的主要贡献和技术创新点"
        
        return requirement
        
    except Exception as e:
        return f"根据论文《{poster_name}》的核心内容，创建一个动态演示动画，要讲清楚论文的主要贡献和技术创新点"


def get_target_figure_image(args) -> str:
    """获取相应figure图片的信息"""
    try:
        from utils.path_utils import get_paper_name_from_path, load_json_file
        
        paper_name = get_paper_name_from_path(args.poster_path)
        
        # 加载图片信息
        try:
            images = load_json_file('images_and_tables', paper_name, f'{args.poster_name}_images.json', 'qwen-2.5-vl-7b', '4o')
            
            if images:
                # 选择第一个图片作为目标图片
                first_image_id = list(images.keys())[0]
                first_image = images[first_image_id]
                
                image_info = f"""
目标Figure图片信息:
- 图片ID: {first_image_id}
- 图片路径: {first_image.get('image_path', '')}
- 图片标题: {first_image.get('caption', '')}
- 图片描述: {first_image.get('description', '')}

图片详细信息:
{json.dumps(first_image, indent=2, ensure_ascii=False)}

注意: 请根据这个图片的实际内容和结构来设计Manim动画，确保动画能够准确反映图片中的关键元素和布局。
"""
                return image_info
            else:
                return "未找到相应的figure图片信息，请根据JSON规划内容生成通用的动画代码。"
                
        except Exception as e:
            return f"图片信息加载失败: {e}，请根据JSON规划内容生成通用的动画代码。"
            
    except Exception as e:
        return f"获取图片信息时出错: {e}，请根据JSON规划内容生成通用的动画代码。"


def classify_section_type(title: str) -> str:
    """根据标题分类章节类型"""
    title_lower = title.lower()
    
    if any(keyword in title_lower for keyword in ['abstract', '摘要']):
        return 'Abstract'
    elif any(keyword in title_lower for keyword in ['introduction', '引言', '综述']):
        return 'Introduction'
    elif any(keyword in title_lower for keyword in ['method', 'approach', '方法', '算法']):
        return 'Method'
    elif any(keyword in title_lower for keyword in ['experiment', 'result', '实验', '结果']):
        return 'Experiment'
    elif any(keyword in title_lower for keyword in ['conclusion', '结论', '总结']):
        return 'Conclusion'
    elif any(keyword in title_lower for keyword in ['related', '相关工作']):
        return 'Related Work'
    elif any(keyword in title_lower for keyword in ['implementation', '实现']):
        return 'Implementation'
    else:
        return 'Other'


def select_target_figure(raw_content: Dict[str, Any], args) -> str:
    """智能选择用于动画的目标图表并提供详细信息"""
    try:
        # 获取论文名称用于找到图表文件
        from utils.path_utils import get_paper_name_from_path, load_json_file
        
        paper_name = get_paper_name_from_path(args.poster_path)
        
        # 尝试加载图表和表格信息
        try:
            images = load_json_file('images_and_tables', paper_name, f'{args.poster_name}_images.json', 'qwen-2.5-vl-7b', '4o')
            tables = load_json_file('images_and_tables', paper_name, f'{args.poster_name}_tables.json', 'qwen-2.5-vl-7b', '4o')
        except:
            # 如果加载失败，返回通用描述
            return "目标图表: 论文中的核心方法架构图或算法流程图"
        
        # 分析图表内容并选择最适合的
        figure_analysis = {
            'total_images': len(images) if images else 0,
            'total_tables': len(tables) if tables else 0,
            'selected_figures': []
        }
        
        # 优先选择架构图、流程图等适合动画的图片
        priority_keywords = ['architecture', 'framework', 'pipeline', 'flow', 'diagram', 'model', 'system']
        
        if images:
            for img_id, img_info in images.items():
                img_path = img_info.get('image_path', '')
                caption = img_info.get('caption', '').lower()
                
                # 计算优先级分数
                priority_score = 0
                for keyword in priority_keywords:
                    if keyword in caption:
                        priority_score += 1
                
                figure_info = {
                    'id': img_id,
                    'path': img_path,
                    'caption': img_info.get('caption', ''),
                    'priority_score': priority_score,
                    'type': 'image'
                }
                figure_analysis['selected_figures'].append(figure_info)
        
        # 也考虑表格，尤其是结果表
        if tables:
            for table_id, table_info in tables.items():
                table_path = table_info.get('table_path', '')
                caption = table_info.get('caption', '').lower()
                
                # 表格的优先级较低，但结果表可以考虑
                priority_score = 0
                if any(keyword in caption for keyword in ['result', 'performance', 'comparison']):
                    priority_score = 0.5
                
                figure_info = {
                    'id': table_id,
                    'path': table_path,
                    'caption': table_info.get('caption', ''),
                    'priority_score': priority_score,
                    'type': 'table'
                }
                figure_analysis['selected_figures'].append(figure_info)
        
        # 按优先级排序
        figure_analysis['selected_figures'].sort(key=lambda x: x['priority_score'], reverse=True)
        
        # 生成详细的图表信息描述
        if figure_analysis['selected_figures']:
            top_figure = figure_analysis['selected_figures'][0]
            
            figure_description = f"""
图表分析结果:
- 总图片数: {figure_analysis['total_images']}
- 总表格数: {figure_analysis['total_tables']}

推荐目标图表:
- ID: {top_figure['id']}
- 类型: {top_figure['type']}
- 路径: {top_figure['path']}
- 标题: {top_figure['caption']}
- 优先级分数: {top_figure['priority_score']}

其他可用图表:
"""
            
            for i, fig in enumerate(figure_analysis['selected_figures'][1:5], 1):  # 显示前5个
                figure_description += f"- {i}. {fig['type']}: {fig['caption'][:100]}...\n"
            
            return figure_description
        else:
            return "目标图表: 论文中的核心方法架构图或算法流程图（未找到具体图表文件）"
        
    except Exception as e:
        return f"图表分析错误: {e}\n默认目标: 论文中的核心方法架构图或算法流程图"


def sanitize_class_name(poster_name: str) -> str:
    """清理类名，确保符合Python命名规范"""
    # 移除特殊字符，只保留字母和数字
    import re
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', poster_name)
    # 确保以字母开头
    if clean_name and clean_name[0].isdigit():
        clean_name = 'Paper' + clean_name
    elif not clean_name:
        clean_name = 'PaperAnimation'
    return clean_name


def extract_python_code(response_text: str) -> str:
    """从响应中提取Python代码"""
    import re
    
    # 尝试提取代码块
    code_patterns = [
        r'```python\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'from manim import \*(.*?)(?=\n\n|\Z)',
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            if 'from manim import' in code and 'class' in code:
                return code
    
    # 如果没有找到代码块，返回整个响应（可能是纯代码）
    if 'from manim import' in response_text and 'class' in response_text:
        return response_text.strip()
    
    return ""


def validate_manim_code(code: str) -> bool:
    """验证生成的Manim代码基本语法和API正确性"""
    try:
        # 基本检查
        required_elements = [
            'from manim import',
            'class',
            'Scene',
            'def construct',
        ]
        
        for element in required_elements:
            if element not in code:
                print(f"⚠️ 代码验证失败: 缺少 '{element}'")
                return False
        
        # 检查是否使用了不存在的Manim类
        invalid_manim_classes = [
            'Diamond',  # 不存在的形状
            'Triangle',  # 应该使用 Polygon
            'Hexagon',   # 应该使用 RegularPolygon
            'Pentagon',  # 应该使用 RegularPolygon
        ]
        
        for invalid_class in invalid_manim_classes:
            if invalid_class in code:
                print(f"⚠️ 代码验证失败: 使用了不存在的Manim类 '{invalid_class}'")
                print(f"建议使用: RegularPolygon 或其他标准形状")
                return False
        
        # 尝试编译检查
        compile(code, '<string>', 'exec')
        return True
        
    except SyntaxError as e:
        print(f"⚠️ 代码语法错误: {e}")
        return False
    except Exception as e:
        print(f"⚠️ 代码验证失败: {e}")
        return False


def create_fallback_planning(poster_name: str) -> Dict[str, Any]:
    """创建默认的动画规划结构"""
    return {
        "paper_analysis": {
            "research_domain": "academic research",
            "core_method": "research methodology",
            "figure_purpose": "illustrate key concepts",
            "technical_complexity": "medium"
        },
        "animation_design": {
            "total_duration": "60",
            "scene_sequence": [
                {
                    "scene_id": 1,
                    "name": "Introduction",
                    "duration": "20",
                    "content_focus": "Paper overview",
                    "animation_elements": [
                        {
                            "target": "title",
                            "action": "Write",
                            "timing": "0",
                            "duration": "3"
                        }
                    ]
                }
            ]
        }
    }


def create_default_manim_script(poster_name: str) -> str:
    """创建默认的Manim动画脚本"""
    class_name = sanitize_class_name(poster_name)
    
    return f'''from manim import *
import numpy as np

class {class_name}Animation(Scene):
    def construct(self):
        # 场景配置
        self.camera.background_color = WHITE
        
        # 创建标题
        title = Text("{poster_name}", font_size=48, color=BLACK)
        title.to_edge(UP)
        
        # 创建主要内容
        content = VGroup(
            Text("学术论文动画演示", font_size=36, color=DARK_BLUE),
            Text("核心方法与贡献", font_size=24, color=DARK_GRAY),
        ).arrange(DOWN, buff=0.5)
        content.next_to(title, DOWN, buff=1)
        
        # 动画序列
        self.play(Write(title), run_time=2)
        self.wait(1)
        
        self.play(FadeIn(content), run_time=2)
        self.wait(2)
        
        # 创建框架图
        framework = VGroup(
            Rectangle(width=3, height=2, color=BLUE),
            Text("核心算法", font_size=20, color=BLUE)
        )
        framework.next_to(content, DOWN, buff=1)
        
        self.play(DrawBorderThenFill(framework[0]), run_time=1.5)
        self.play(Write(framework[1]), run_time=1)
        self.wait(3)
        
        # 结束
        self.play(FadeOut(VGroup(title, content, framework)), run_time=2)
        self.wait(1)
'''


def create_default_manim_script_file(script_path: str, poster_name: str):
    """创建默认的Manim脚本文件"""
    script_content = create_default_manim_script(poster_name)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ 默认Manim脚本已创建: {script_path}")


def analyze_research_context(raw_content: Dict[str, Any]) -> Dict[str, str]:
    """分析研究背景和领域"""
    try:
        sections = raw_content.get('sections', [])
        
        # 分析研究领域
        domain_keywords = {
            'deep_learning': ['deep learning', 'neural network', 'cnn', 'rnn', 'transformer'],
            'nlp': ['natural language', 'nlp', 'text', 'language model', 'bert'],
            'computer_vision': ['computer vision', 'image', 'visual', 'object detection'],
            'machine_learning': ['machine learning', 'ml', 'algorithm', 'model'],
            'ai': ['artificial intelligence', 'ai', 'intelligent'],
            'data_science': ['data', 'dataset', 'analysis', 'mining'],
            'robotics': ['robot', 'robotic', 'autonomous'],
            'other': []
        }
        
        all_text = ' '.join([section.get('content', '') for section in sections]).lower()
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(all_text.count(keyword) for keyword in keywords)
            domain_scores[domain] = score
        
        # 找到最匹配的领域
        primary_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[primary_domain] == 0:
            primary_domain = 'general_research'
        
        # 分析研究动机和贡献
        motivation_analysis = ""
        contribution_analysis = ""
        
        for section in sections:
            title = section.get('title', '').lower()
            content = section.get('content', '')
            
            if 'introduction' in title or 'abstract' in title:
                motivation_analysis += f"\n{content[:500]}..."
            elif 'conclusion' in title or 'contribution' in title:
                contribution_analysis += f"\n{content[:500]}..."
        
        return {
            'domain': primary_domain,
            'analysis': f"""
研究领域: {primary_domain}
研究动机: {motivation_analysis.strip()}
主要贡献: {contribution_analysis.strip()}
"""
        }
        
    except Exception as e:
        return {
            'domain': 'unknown',
            'analysis': f'研究背景分析错误: {e}'
        }


def assess_technical_complexity(raw_content: Dict[str, Any]) -> str:
    """评估技术复杂度"""
    try:
        sections = raw_content.get('sections', [])
        all_text = ' '.join([section.get('content', '') for section in sections]).lower()
        
        # 复杂度指标
        complexity_indicators = {
            'mathematical': ['equation', 'formula', 'theorem', 'proof', 'optimization'],
            'algorithmic': ['algorithm', 'complexity', 'time complexity', 'space complexity'],
            'experimental': ['experiment', 'dataset', 'baseline', 'evaluation'],
            'theoretical': ['theoretical', 'theory', 'framework', 'model']
        }
        
        scores = {}
        for category, keywords in complexity_indicators.items():
            score = sum(all_text.count(keyword) for keyword in keywords)
            scores[category] = score
        
        total_score = sum(scores.values())
        
        if total_score > 20:
            return 'very_complex'
        elif total_score > 10:
            return 'complex'
        elif total_score > 5:
            return 'medium'
        else:
            return 'simple'
            
    except Exception as e:
        return 'unknown'