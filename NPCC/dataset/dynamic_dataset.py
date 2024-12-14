import os
import torch
from torch.utils.data import Dataset
from .preprocess import load_and_normalize_point_cloud
import numpy as np  # Add this line


class DynamicPointCloudDataset(Dataset):
    def __init__(self, data_dir, model_height=1.8, voxel_resolution=1024, num_points=1024, sequence_length=5):
        """
        Initialize the dynamic point cloud dataset.
        :param data_dir: Root directory of the dataset, where each sequence should be in a separate subdirectory.
        :param sequence_length: Number of frames to load at a time.
        """
        self.data_dir = data_dir
        print(f"Dataset path: {os.path.abspath(self.data_dir)}")  # Print the absolute path

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

        # Load point clouds from the sequence
        sequence_data = []
        for frame_path in frames[:self.sequence_length]:
            _, points = load_and_normalize_point_cloud(
                frame_path,
                model_height=self.model_height,
                voxel_resolution=self.voxel_resolution,
                num_points=self.num_points
            )
            sequence_data.append(points)

        # Convert the list to a NumPy array and then to a tensor
        sequence_data = np.array(sequence_data)
        return torch.tensor(sequence_data, dtype=torch.float32)
