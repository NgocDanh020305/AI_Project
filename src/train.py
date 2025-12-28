import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data import IAMDataset, collate_fn
from src.model import CRNN
from src.utils import TextEncoder

# --- CẤU HÌNH ---
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.001
IMG_HEIGHT = 32
IMG_WIDTH = 128
HIDDEN_SIZE = 256
DATA_DIR = "./data/words"  # Folder chứa các subfolder ảnh
LIST_FILE = "./data/words.txt"  # File metadata
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    # 1. Dataset & DataLoader
    if not os.path.exists(LIST_FILE):
        print(f"Lỗi: Không tìm thấy file {LIST_FILE}. Hãy tải IAM dataset trước.")
        return

    full_dataset = IAMDataset(DATA_DIR, LIST_FILE, IMG_HEIGHT, IMG_WIDTH)

    # Chia train/val (90% - 10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model, Loss, Optimizer
    encoder = TextEncoder()
    n_class = len(encoder)  # vocab + blank

    model = CRNN(IMG_HEIGHT, 1, n_class, HIDDEN_SIZE).to(DEVICE)

    # CTC Loss: blank=0
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Training Loop
    best_loss = float('inf')
    os.makedirs("saved_models", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, targets, target_lengths, _ in pbar:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            # Forward
            preds = model(images)  # (T, B, C)

            # Tính Input Lengths cho CTC (độ dài chuỗi time-step sau CNN)
            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=preds.size(0), dtype=torch.long).to(DEVICE)

            # CTC Loss expects log_probs
            log_probs = torch.nn.functional.log_softmax(preds, dim=2)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping để tránh bùng nổ gradient trong RNN
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "saved_models/best_model.pth")
            print("Model saved!")


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets, target_lengths, _ in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            target_lengths = target_lengths.to(DEVICE)

            preds = model(images)
            batch_size = images.size(0)
            input_lengths = torch.full(size=(batch_size,), fill_value=preds.size(0), dtype=torch.long).to(DEVICE)
            log_probs = torch.nn.functional.log_softmax(preds, dim=2)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            val_loss += loss.item()
    return val_loss / len(loader)


if __name__ == "__main__":
    train()