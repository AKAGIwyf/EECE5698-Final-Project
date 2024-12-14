import torch.nn as nn
import torch

class DynamicPointCloudAutoencoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=32, sequence_length=5):
        """
        Dynamic Point Cloud Autoencoder
        :param input_dim: Dimension of input points (default: 3)
        :param hidden_dim: Dimension of hidden layer features
        :param latent_dim: Dimension of latent features
        :param sequence_length: Number of frames in the dynamic point cloud
        """
        super(DynamicPointCloudAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, latent_dim)
        )

        # LSTM for temporal modeling
        self.temporal_model = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            batch_first=True
        )

        # Decoder
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

        # Flatten each frame of the point cloud for input to the encoder
        x = x.view(-1, input_dim)  # (batch_size * sequence_length * num_points, input_dim)

        # Encode each frame of the point cloud
        latent = self.encoder(x)  # (batch_size * sequence_length * num_points, latent_dim)

        # Reshape latent features back to frame shape
        latent = latent.view(batch_size, sequence_length, num_points,
                             -1)  # (batch_size, sequence_length, num_points, latent_dim)

        # Aggregate point cloud features for temporal modeling
        latent = torch.mean(latent, dim=2)  # (batch_size, sequence_length, latent_dim)

        # Perform temporal modeling with LSTM
        latent, _ = self.temporal_model(latent)  # (batch_size, sequence_length, latent_dim)

        # Restore point cloud features
        latent = latent.unsqueeze(2).repeat(1, 1, num_points,
                                            1)  # (batch_size, sequence_length, num_points, latent_dim)
        latent = latent.view(-1, latent.size(-1))  # Flatten to (batch_size * sequence_length * num_points, latent_dim)

        # Decode reconstructed point clouds for each frame
        reconstructed = self.decoder(latent)  # (batch_size * sequence_length * num_points, input_dim)

        # Reshape back to dynamic point cloud shape
        reconstructed = reconstructed.view(batch_size, sequence_length, num_points, input_dim)

        return latent, reconstructed
