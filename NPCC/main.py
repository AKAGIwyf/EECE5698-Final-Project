import torch
from torch.utils.data import DataLoader
from dataset.dynamic_dataset import DynamicPointCloudDataset
from models.dynamic_autoencoder import DynamicPointCloudAutoencoder
from train import train


def main():
    # Parameter settings
    data_dir = "./dataset/exercise_vox11_organized"  # Path to the dynamic point cloud dataset
    batch_size = 4
    sequence_length = 10
    model_height = 1.8
    voxel_resolution = 1024
    num_epochs = 20
    learning_rate = 1e-2

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the dynamic point cloud dataset
    print("Loading dynamic point cloud dataset...")
    dataset = DynamicPointCloudDataset(
        data_dir=data_dir,
        model_height=model_height,
        voxel_resolution=voxel_resolution,
        sequence_length=sequence_length
    )
    print(f"Dataset size: {len(dataset)}")  # Print dataset size
    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check the data path and file contents.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(dataset)}")

    # Initialize the model
    print("Initializing model...")
    model = DynamicPointCloudAutoencoder(sequence_length=sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    print("Starting training...")
    train(dataloader, model, optimizer, num_epochs=num_epochs, device=device)


if __name__ == "__main__":
    main()
