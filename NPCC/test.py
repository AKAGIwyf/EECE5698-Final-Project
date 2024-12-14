import torch
from torch.utils.data import DataLoader
from models.dynamic_autoencoder import DynamicPointCloudAutoencoder
from dataset.dynamic_dataset import DynamicPointCloudDataset

def test(dataloader, model, device="cuda"):
    """
    Test the model and compute the reconstruction error.
    :param dataloader: DataLoader for the test dataset
    :param model: Trained model
    :param device: Device to use (default: "cuda")
    """
    model.eval()
    model.to(device)
    total_loss = 0

    with torch.no_grad():
        for batch_idx, sequence_data in enumerate(dataloader):
            sequence_data = sequence_data.to(device)  # Shape: (batch_size, sequence_length, num_points, 3)
            latent, reconstructed = model(sequence_data)  # Forward pass

            # Compute Mean Squared Error (MSE) loss
            loss = torch.nn.functional.mse_loss(reconstructed, sequence_data)
            total_loss += loss.item()

            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Average reconstruction error on the test dataset: {avg_loss:.6f}")
    return avg_loss


if __name__ == "__main__":
    # Configuration parameters
    data_dir = "./dataset/exercise_vox11_organized"  # Path to the dynamic point cloud dataset
    batch_size = 4
    sequence_length = 10
    model_height = 1.8
    voxel_resolution = 1024
    num_points = 1024

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the test dataset
    print("Loading test dataset...")
    dataset = DynamicPointCloudDataset(
        data_dir=data_dir,
        model_height=model_height,
        voxel_resolution=voxel_resolution,
        num_points=num_points,
        sequence_length=sequence_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load the trained model
    print("Loading model...")
    model = DynamicPointCloudAutoencoder(
        input_dim=3,
        hidden_dim=64,
        latent_dim=32,
        sequence_length=sequence_length
    )
    model.load_state_dict(torch.load("./model/saved_model.pth"))
    print("Model loaded successfully!")

    # Test the model
    print("Starting testing...")
    avg_loss = test(dataloader, model, device=device)
    print(f"Testing completed! Average loss: {avg_loss:.6f}")
