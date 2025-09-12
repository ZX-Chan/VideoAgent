import os
import json
from typing import Dict, Any

def get_paper_name_from_path(poster_path: str) -> str:
    """从poster_path中提取论文名称"""
    path_parts = poster_path.split('/')
    # 查找包含论文名称的部分（通常是倒数第二个部分）
    for i in range(len(path_parts) - 1):
        if path_parts[i] == 'data' and i + 1 < len(path_parts):
            return path_parts[i + 1]
    # 如果找不到data目录，则使用文件名（去掉.pdf）
    return os.path.basename(poster_path).replace('.pdf', '')

def create_output_dirs(model_name_t: str, model_name_v: str, paper_name: str) -> Dict[str, str]:
    """创建统一的输出目录结构"""
    base_dir = f'<{model_name_t}_{model_name_v}>'
    
    dirs = {
        'contents': f'{base_dir}_contents/{paper_name}/',
        'images_and_tables': f'{base_dir}_images_and_tables/{paper_name}/',
        'tree_splits': f'{base_dir}_tree_splits/{paper_name}/',
        'equations': f'{base_dir}_equations/{paper_name}/',
        'tts': f'{base_dir}_tts/{paper_name}/',
        'generated_posters': f'{base_dir}_generated_posters/{paper_name}/',
        'temp': f'{base_dir}_temp/{paper_name}/'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_file_path(dir_type: str, paper_name: str, filename: str, model_name_t: str, model_name_v: str) -> str:
    """获取指定类型目录下的文件路径"""
    dirs = create_output_dirs(model_name_t, model_name_v, paper_name)
    return os.path.join(dirs[dir_type], filename)

def save_json_file(data: Dict[str, Any], dir_type: str, paper_name: str, filename: str, 
                  model_name_t: str, model_name_v: str) -> str:
    """保存JSON文件到指定目录"""
    file_path = get_file_path(dir_type, paper_name, filename, model_name_t, model_name_v)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return file_path

def load_json_file(dir_type: str, paper_name: str, filename: str, 
                  model_name_t: str, model_name_v: str) -> Dict[str, Any]:
    """从指定目录加载JSON文件"""
    file_path = get_file_path(dir_type, paper_name, filename, model_name_t, model_name_v)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def file_exists(dir_type: str, paper_name: str, filename: str, 
               model_name_t: str, model_name_v: str) -> bool:
    """检查文件是否存在"""
    file_path = get_file_path(dir_type, paper_name, filename, model_name_t, model_name_v)
    return os.path.exists(file_path) 