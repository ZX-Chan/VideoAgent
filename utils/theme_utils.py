# -*- coding: utf-8 -*-
"""
PPT主题样式工具函数
用于加载和应用PPT主题配置
"""

import json
import os
from typing import Dict, Any, Tuple


def load_theme_config(config_path: str = "config/ppt_theme.json") -> Dict[str, Any]:
    """
    加载PPT主题配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        主题配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"主题配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_theme_to_content(content: Dict[str, Any], theme_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    将主题样式应用到内容
    
    Args:
        content: 内容字典
        theme_config: 主题配置
        
    Returns:
        应用样式后的内容
    """
    theme = theme_config.get('theme', {})
    
    # 应用标题样式
    if 'title' in content:
        title_style = theme.get('title', {})
        content['title'] = apply_text_style(content['title'], title_style)
    
    # 应用内容样式
    for key in ['textbox1', 'textbox2']:
        if key in content:
            content_style = theme.get('content', {})
            content[key] = apply_text_style(content[key], content_style)
    
    return content


def apply_text_style(text_data: list, style: Dict[str, Any]) -> list:
    """
    应用文本样式到文本数据
    
    Args:
        text_data: 文本数据列表
        style: 样式配置
        
    Returns:
        应用样式后的文本数据
    """
    if not isinstance(text_data, list):
        return text_data
    
    styled_data = []
    for item in text_data:
        if isinstance(item, dict) and 'runs' in item:
            # 应用样式到runs
            for run in item['runs']:
                if 'text_color' in style:
                    run['color'] = tuple(style['text_color'])
                if 'fill_color' in style:
                    run['fill_color'] = tuple(style['fill_color'])
                if 'font_size' in style:
                    run['font_size'] = style['font_size']
                if 'font_family' in style:
                    run['font_family'] = style['font_family']
                if 'font_weight' in style:
                    run['font_weight'] = style['font_weight']
        
        styled_data.append(item)
    
    return styled_data


def get_theme_colors(theme_config: Dict[str, Any]) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
    """
    从主题配置中获取颜色
    
    Args:
        theme_config: 主题配置
        
    Returns:
        (标题文字颜色, 内容样式字典)
    """
    theme = theme_config.get('theme', {})
    title_style = theme.get('title', {})
    content_style = theme.get('content', {})
    
    title_text_color = tuple(title_style.get('text_color', [128, 0, 128]))
    
    return title_text_color, content_style


def get_slide_dimensions(theme_config: Dict[str, Any]) -> Tuple[float, float]:
    """
    从主题配置中获取幻灯片尺寸
    
    Args:
        theme_config: 主题配置
        
    Returns:
        (宽度英寸, 高度英寸)
    """
    theme = theme_config.get('theme', {})
    dimensions = theme.get('slide_dimensions', {})
    
    width = dimensions.get('width_inches', 16)
    height = dimensions.get('height_inches', 9)
    
    return width, height


def get_layout_config(theme_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    从主题配置中获取布局配置
    
    Args:
        theme_config: 主题配置
        
    Returns:
        布局配置字典
    """
    theme = theme_config.get('theme', {})
    return theme.get('layout', {})


def create_theme_dict(theme_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建兼容旧代码的主题字典
    
    Args:
        theme_config: 主题配置
        
    Returns:
        主题字典
    """
    theme = theme_config.get('theme', {})
    layout = theme.get('layout', {})
    
    return {
        'panel_visible': layout.get('panel_visible', True),
        'textbox_visible': layout.get('textbox_visible', False),
        'figure_visible': layout.get('figure_visible', False),
        'panel_theme': layout.get('panel_theme', {}),
        'textbox_theme': layout.get('textbox_theme'),
        'figure_theme': layout.get('figure_theme')
    } 