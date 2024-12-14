import torch
from visualize import visualize_point_cloud  # 从 visualize.py 导入可视化函数


def test(dataloader, model, device="cuda"):
    """
    测试模型，并可视化原始点云和重构点云。
    :param dataloader: 测试集 DataLoader
    :param model: 训练好的模型
    :param device: 设备（默认为 "cuda"）
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch_idx, sequence_data in enumerate(dataloader):
            sequence_data = sequence_data.to(device)  # 形状 (batch_size, sequence_length, num_points, 3)
            latent, reconstructed = model(sequence_data)  # 前向传播

            # 可视化第一个样本的原始点云和重构点云
            batch_size, sequence_length, num_points, input_dim = sequence_data.shape
            for seq_idx in range(sequence_length):
                original_frame = sequence_data[0, seq_idx].cpu().numpy()  # 获取第一个样本的第 seq_idx 帧
                reconstructed_frame = reconstructed[0, seq_idx].cpu().numpy()

                print(f"Visualizing sequence {seq_idx + 1}/{sequence_length} for batch {batch_idx + 1}")
                visualize_point_cloud(original_frame, title=f"Original Point Cloud (Frame {seq_idx + 1})")
                visualize_point_cloud(reconstructed_frame, title=f"Reconstructed Point Cloud (Frame {seq_idx + 1})")

            break  # 只可视化一个 batch

if __name__ == "__main__":
    from models.dynamic_autoencoder import DynamicPointCloudAutoencoder
    from dataset.dynamic_dataset import DynamicPointCloudDataset
    from torch.utils.data import DataLoader

    # 配置参数
    data_dir = "./dataset/exercise_vox11"
    batch_size = 4
    sequence_length = 10
    model_height = 1.8
    voxel_resolution = 1024
    num_points = 1024

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载测试数据集
    print("加载测试数据集...")
    dataset = DynamicPointCloudDataset(
        data_dir=data_dir,
        model_height=model_height,
        voxel_resolution=voxel_resolution,
        num_points=num_points,
        sequence_length=sequence_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 加载训练好的模型
    print("加载模型...")
    model = DynamicPointCloudAutoencoder(
        input_dim=3,
        hidden_dim=64,
        latent_dim=32,
        sequence_length=sequence_length
    )
    model.load_state_dict(torch.load("./model/saved_model.pth"))
    print("模型加载完成！")

    # 测试模型并可视化结果
    print("开始测试并可视化结果...")
    test(dataloader, model, device=device)
