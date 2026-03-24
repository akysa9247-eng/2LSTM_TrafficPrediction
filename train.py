# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import config
import numpy as np

def train_model(model, train_X, train_Y, val_X, val_Y):
    """训练模型并返回训练好的模型"""
    # 创建 DataLoader
    train_dataset = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_Y, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_Y, dtype=torch.float32)
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)


    print("开始训练...")
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * batch_X.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                total_val_loss += loss.item() * batch_X.size(0)
                all_val_preds.append(outputs.cpu().numpy())
                all_val_targets.append(batch_Y.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        val_preds = np.concatenate(all_val_preds, axis=0)
        val_targets = np.concatenate(all_val_targets, axis=0)
        val_rmse = np.sqrt(np.mean((val_targets - val_preds) ** 2))

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | LR: {current_lr:.6f} | "
                  f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                  f"Val RMSE: {val_rmse:.6f}")

    return model