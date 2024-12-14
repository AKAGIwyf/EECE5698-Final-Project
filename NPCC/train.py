import torch


def train(dataloader, model, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_loss = float("inf")  # 用于保存最优模型
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, sequence_data in enumerate(dataloader):
            sequence_data = sequence_data.to(device)  # (B, T, N, C)

            print(f"Batch {batch_idx + 1}: Input to model shape: {sequence_data.shape}")

            # 前向传播
            optimizer.zero_grad()
            latent, reconstructed = model(sequence_data)

            # 计算损失
            loss = torch.nn.functional.mse_loss(reconstructed, sequence_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("保存最优模型...")
            torch.save(model.state_dict(), "./model/saved_model.pth")  # 保存模型

