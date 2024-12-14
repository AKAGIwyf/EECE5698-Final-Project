import torch


def train(dataloader, model, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_loss = float("inf")  # To store the best model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, sequence_data in enumerate(dataloader):
            sequence_data = sequence_data.to(device)  # Shape: (B, T, N, C)

            # Forward pass
            optimizer.zero_grad()
            latent, reconstructed = model(sequence_data)

            # Compute loss
            loss = torch.nn.functional.mse_loss(reconstructed, sequence_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("Saving the best model...")
            torch.save(model.state_dict(), "./model/saved_model.pth")  # Save the model
