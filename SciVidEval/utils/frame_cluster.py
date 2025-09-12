
import os
import torch
import clip
import json
import subprocess
import h5py
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from pathlib import Path
from tqdm import tqdm
import math
import re

def get_video_duration(input_path):
    """
    使用 ffprobe 获取视频时长（秒）。
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise Exception(f"ffprobe 错误: {result.stderr}")
    duration_str = result.stdout.strip()
    try:
        duration = float(duration_str)
        return duration
    except ValueError:
        raise Exception(f"无法解析时长: {duration_str}")


def extract_frames(input_path, frame_dir, interval, duration):
    """
    使用 ffmpeg 每隔 interval 秒提取一帧图片，并返回时间戳列表（格式：HH:MM:SS.mmm）。

    参数：
        input_path (str or Path): 输入视频文件路径
        frame_dir (str or Path): 保存提取帧的目标目录
        interval (int or float): 每隔多少秒提取一帧
        duration (float): 视频总时长（秒）

    返回：
        List[str]: 时间戳字符串列表，格式形如 ["00:00:00.000", "00:00:10.000", ...]
    """
    # 确保输出目录存在
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    timestamps = []
    total_frames = math.ceil(duration / interval)

    for i in tqdm(range(total_frames)):
        # 当前帧对应的时间（秒）
        timestamp_sec = i * interval

        # 如果超过视频总时长，就停止提取
        if timestamp_sec > duration:
            break

        # 将时间转换为 HH:MM:SS.mmm 格式
        hrs = int(timestamp_sec // 3600)
        mins = int((timestamp_sec % 3600) // 60)
        secs = timestamp_sec % 60
        # 形如 00:00:10.000
        timestamp_str = f"{hrs:02}:{mins:02}:{secs:06.3f}"



        # 生成输出帧文件名
        frame_filename = f"frame_{i + 1:04}.jpg"
        frame_path = frame_dir / frame_filename

        # 将该时间戳加入列表
        timestamps.append({
            "path": frame_filename,
            "timestamp": timestamp_str})

        # 调用 ffmpeg 提取指定时间点的帧
        cmd = [
            "ffmpeg",
            "-ss", timestamp_str,  # 跳转到对应时间
            "-i", str(input_path),
            "-frames:v", "1",  # 只要1帧
            "-q:v", "2",  # 图片质量，2~31，数值越小质量越高
            "-y",  # 覆盖输出文件
            str(frame_path)
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"提取帧失败：时间戳 {timestamp_str}，错误信息：{result.stderr}")

    return timestamps



def extract_clip_features(image_paths, model, processor, video_name, device='cpu'):
    """
    给定图片路径列表，通过 CLIP 提取图像特征，并将特征存储到 ./temp/{video_name}/feature.ht 文件中；
    同时将 image_path 与其对应的特征索引 ID 存到一个 JSON 文件 ./temp/{video_name}/feature_map.json。

    参数:
    - image_paths: list[str], 图片路径列表
    - model: CLIP 模型 (例如 clip.load(...) 后的 model)
    - processor: 对应的图像处理函数 (例如 clip.load(...) 返回的 preprocess，或者 HF 的 CLIPProcessor)
    - video_name: 保存文件使用的视频名，可自定义
    - device: 运行设备，'cpu' 或 'cuda'
    """
    h5_path = f"./temp/{video_name}/feature.ht"
    json_path = f"./temp/{video_name}/feature_map.json"

    if os.path.exists(h5_path) and os.path.isfile(h5_path):
        all_features = h5py.File(h5_path, 'r')["features"]
        with open(json_path, 'r') as f:
            feature_mapping = json.load(f)
        print(f"已加载视频特征: {h5_path}")
        return all_features, feature_mapping
    else:
        os.makedirs(f"./temp/{video_name}", exist_ok=True)



    all_features = []
    # 临时存储 (image_path, feature_id) 的映射
    feature_mapping = {}

    for idx, img_path in enumerate(image_paths):
        # 1. 读图并处理
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # 2. 前向计算：提取特征
            image_features = model.encode_image(inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        # 3. 收集特征
        all_features.append(image_features.cpu().numpy().squeeze())

        # 4. 记录该图对应的特征 ID
        feature_mapping[idx] = img_path

    # 堆叠成 (num_images, feature_dim)
    all_features = np.vstack(all_features)

    # 5. 将所有特征写入一个 HDF5 文件
    with h5py.File(h5_path, "w") as f:
        # 创建一个名为 "features" 的 dataset，并一次性写入
        f.create_dataset("features", data=all_features, compression="gzip")

    # 6. 将 image_paths 与 feature_id 的映射存成 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(feature_mapping, f, ensure_ascii=False, indent=4)

    print(f"所有特征已写入: {h5_path}")
    print(f"映射信息已写入: {json_path}")

    # 如果仍需要在 Python 中返回特征数组，可在此 return
    return all_features, feature_mapping

def cluster_and_select_keyframes_dbscan(features, image_paths, eps=0.5, min_samples=5):
    """
    使用 DBSCAN 对特征进行聚类，并在每个聚类中选取一个关键帧（距离簇中心最近）。
    返回关键帧对应的图片路径列表。

    参数：
    - feature: (num_images, feature_dim) 的特征矩阵
    - image_paths: 对应的图片路径列表
    - eps: DBSCAN 的邻域半径
    - min_samples: DBSCAN 中将一个点视为核心点所需的最小样本数
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)  # shape: (num_images, )

    # 去除噪声标签 -1
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    keyframes = []
    # for cluster_id in unique_labels:
    #     # 找到该簇对应的所有索引
    #     cluster_indices = np.where(labels == cluster_id)[0]
    #     if len(cluster_indices) == 0:
    #         continue
    #
    #     # 获取该簇的所有特征向量
    #     cluster_features = features[cluster_indices]
    #     # 求该簇的均值向量，作为“类中心”
    #     center = np.mean(cluster_features, axis=0)
    #     # 计算每个点到类中心的距离
    #     distances = np.linalg.norm(cluster_features - center, axis=1)
    #     # 找到距离类中心最近的那个索引
    #     min_dist_index = cluster_indices[np.argmin(distances)]
    #     keyframes.append(image_paths[min_dist_index])

    for cluster_id in unique_labels:
        # 找到该簇对应的所有索引
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue

        # 获取该簇的所有图像路径
        cluster_image_paths = [image_paths[i] for i in cluster_indices]

        # 定义一个辅助函数，用于从路径中提取帧的编号
        def get_frame_number(path):
            # 从路径中获取文件名，例如 "frame_0001.jpg"
            basename = os.path.basename(path)
            # 使用正则表达式找到文件名中所有的数字序列
            numbers = re.findall(r'\d+', basename)
            # 返回最后一个找到的数字（通常就是帧编号），如果找不到则返回-1
            return int(numbers[-1]) if numbers else -1

        # 使用 max() 函数和自定义的 key，找到编号最大的那个图像路径
        # max() 会对 cluster_image_paths 中的每个元素调用 get_frame_number 函数，
        # 并基于返回的数字来比较元素的大小。
        latest_keyframe = max(cluster_image_paths, key=get_frame_number)

        keyframes.append(latest_keyframe)

    return keyframes


    return keyframes


def process_frames(state):
    folder_path = state['frame_dir']
    video_name = state['video_name']
    # 1. 指定图片所在文件夹
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]


    # 2. 加载 CLIP 模型和处理器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 3. 提取所有图片的特征
    features, feature_mapping = extract_clip_features(image_paths, model, preprocess, video_name, device=device)

    # 4. 用 DBSCAN 聚类，并找出各聚类的关键帧
    #    注意根据你的数据规模与分布，需要调整 eps 和 min_samples
    eps = 0.2
    min_samples = 2
    keyframes = cluster_and_select_keyframes_dbscan(features, image_paths, eps=eps, min_samples=min_samples)

    # 打印聚类后的关键帧
    # print("选出的关键帧：")
    # for idx, kf in enumerate(keyframes):
    #     print(f"Cluster {idx}: {kf}")

    keyframes_set = set(keyframes)

    # 2. 找到非关键帧的路径
    non_keyframes = [path for path in image_paths if path not in keyframes_set]

    # 3. 从磁盘上删除非关键帧对应的图片文件
    for path in non_keyframes:
        if os.path.exists(path):
            os.remove(path)
            # print(f"已删除非关键帧文件: {path}")

    return keyframes

# if __name__ == "__main__":
#     process_frames('./temp/frame/')
