import torch.nn as nn
import torch

class DynamicPointCloudAutoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=32, sequence_length=5):
        """
        动态点云自编码器
        :param input_dim: 输入点的维度 (默认为 3)
        :param hidden_dim: 隐藏层特征维度
        :param latent_dim: 潜在特征维度
        :param sequence_length: 动态点云的帧数
        """
        super(DynamicPointCloudAutoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )

        # LSTM 用于时间序列建模
        self.temporal_model = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        batch_size, sequence_length, num_points, input_dim = x.shape

        # 展平每帧点云以输入编码器
        x = x.view(-1, input_dim)  # (batch_size * sequence_length * num_points, input_dim)

        # 编码每帧点云
        latent = self.encoder(x)  # (batch_size * sequence_length * num_points, latent_dim)

        # 恢复每帧点云的形状
        latent = latent.view(batch_size, sequence_length, num_points,
                             -1)  # (batch_size, sequence_length, num_points, latent_dim)

        # 聚合点云特征用于时间序列建模
        latent = torch.mean(latent, dim=2)  # (batch_size, sequence_length, latent_dim)

        # 使用 LSTM 进行时间序列建模
        latent, _ = self.temporal_model(latent)  # (batch_size, sequence_length, latent_dim)

        # 恢复点云特征
        latent = latent.unsqueeze(2).repeat(1, 1, num_points,
                                            1)  # (batch_size, sequence_length, num_points, latent_dim)
        latent = latent.view(-1, latent.size(-1))  # 展平为 (batch_size * sequence_length * num_points, latent_dim)

        # 解码重构每帧点云
        reconstructed = self.decoder(latent)  # (batch_size * sequence_length * num_points, input_dim)

        # 恢复动态点云的形状
        reconstructed = reconstructed.view(batch_size, sequence_length, num_points, input_dim)

        return latent, reconstructed

