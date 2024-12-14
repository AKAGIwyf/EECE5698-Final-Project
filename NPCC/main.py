import torch
from torch.utils.data import DataLoader
from dataset.dynamic_dataset import DynamicPointCloudDataset
from models.dynamic_autoencoder import DynamicPointCloudAutoencoder
from train import train


def main():
    # 参数设置
    data_dir = "./dataset/exercise_vox11_organized"  # 动态点云数据集路径
    batch_size = 4
    sequence_length = 10
    model_height = 1.8
    voxel_resolution = 1024
    num_epochs = 20
    learning_rate = 1e-2

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载动态点云数据集
    print("加载动态点云数据集...")
    dataset = DynamicPointCloudDataset(
        data_dir=data_dir,
        model_height=model_height,
        voxel_resolution=voxel_resolution,
        sequence_length=sequence_length
    )
    print(f"数据集大小: {len(dataset)}")  # 打印数据集大小
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查数据路径和文件内容。")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"数据集大小: {len(dataset)}")

    # 初始化模型
    print("初始化模型...")
    model = DynamicPointCloudAutoencoder(sequence_length=sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    print("开始训练...")
    train(dataloader, model, optimizer, num_epochs=num_epochs, device=device)


if __name__ == "__main__":
    main()
