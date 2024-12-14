import os
import torch
from torch.utils.data import Dataset
from .preprocess import load_and_normalize_point_cloud
import numpy as np  # 添加此行


class DynamicPointCloudDataset(Dataset):
    def __init__(self, data_dir, model_height=1.8, voxel_resolution=1024, num_points=1024, sequence_length=5):
        """
        初始化动态点云数据集。
        :param data_dir: 数据集根目录，每个序列应在单独的子目录中。
        :param sequence_length: 每次加载的帧数。
        """
        self.data_dir = data_dir
        print(f"数据集路径: {os.path.abspath(self.data_dir)}")  # 打印绝对路径

        self.sequence_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.model_height = model_height
        self.voxel_resolution = voxel_resolution
        self.num_points = num_points
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, idx):
        sequence_dir = self.sequence_dirs[idx]
        frames = sorted([os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir) if f.endswith('.ply')])

        if len(frames) < self.sequence_length:
            raise ValueError(f"Sequence {sequence_dir} has fewer than {self.sequence_length} frames.")

        # 加载序列中的点云
        sequence_data = []
        for frame_path in frames[:self.sequence_length]:
            _, points = load_and_normalize_point_cloud(
                frame_path,
                model_height=self.model_height,
                voxel_resolution=self.voxel_resolution,
                num_points=self.num_points
            )
            sequence_data.append(points)

        # 转换列表为 NumPy 数组，再转换为张量
        sequence_data = np.array(sequence_data)
        print(f"Sequence data shape before conversion: {np.array(sequence_data).shape}")
        return torch.tensor(sequence_data, dtype=torch.float32)
