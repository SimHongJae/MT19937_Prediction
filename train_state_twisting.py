"""
Train State Twisting Network (NCC Group approach)
Goal: Learn MT19937 state twisting relationship
      MT[i] = f(MT[i-624], MT[i-623], MT[i-227])
Input: 65 bits (1 + 32 + 32)
Output: 32 bits
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

from model import StateTwisting


def load_twisting_data(split='train'):
    """Load state twisting dataset"""
    data = np.load(f'data/twisting_{split}.npz')
    X = torch.from_numpy(data['X'])  # State triplets (65 bits)
    y = torch.from_numpy(data['y'])  # Next state (32 bits)
    return X, y


def calculate_bit_accuracy(predictions, targets):
    """
    Calculate bit-wise accuracy
    :param predictions: (batch, 32) probabilities (0-1) from Sigmoid
    :param targets: (batch, 32) ground truth bits
    :return: accuracy (0-1)
    """
    pred_bits = (predictions > 0.5).float()
    correct = (pred_bits == targets).float().mean()
    return correct.item()


def calculate_exact_match(predictions, targets):
    """
    Calculate percentage of samples where all 32 bits match
    :param predictions: (batch, 32) probabilities(0-1) from Sigmoid
    :param targets: (batch, 32) ground truth bits
    :return: exact match rate (0-1)
    """
    pred_bits = (predictions > 0.5).float()
    all_match = (pred_bits == targets).all(dim=1).float().mean()
    return all_match.item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_bit_acc = 0.0
    total_exact = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)

        # Calculate loss (BCE with probabilities)
        loss = criterion(predictions, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate metrics
        bit_acc = calculate_bit_accuracy(predictions, y_batch)
        exact = calculate_exact_match(predictions, y_batch)

        total_loss += loss.item()
        total_bit_acc += bit_acc
        total_exact += exact
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'bit_acc': f'{bit_acc:.4f}',
            'exact': f'{exact:.4f}'
        })

    return total_loss / num_batches, total_bit_acc / num_batches, total_exact / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_bit_acc = 0.0
    total_exact = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            bit_acc = calculate_bit_accuracy(predictions, y_batch)
            exact = calculate_exact_match(predictions, y_batch)

            total_loss += loss.item()
            total_bit_acc += bit_acc
            total_exact += exact
            num_batches += 1

    return total_loss / num_batches, total_bit_acc / num_batches, total_exact / num_batches


def main():
    # Hyperparameters (NCC Group architecture)
    HIDDEN_DIM = 96  # NCC used 96 for 100% accuracy
    BATCH_SIZE = 128  # Smaller batch for 65-bit input
    LEARNING_RATE = 1e-3
    EPOCHS = 50  # May need more epochs for this task
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("STATE TWISTING NETWORK TRAINING (NCC Group)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    X_train, y_train = load_twisting_data('train')
    X_val, y_val = load_twisting_data('val')

    print(f"Train: {len(X_train)} samples, shape {X_train.shape}")
    print(f"Val: {len(X_val)} samples, shape {X_val.shape}")

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = StateTwisting(hidden_dim=HIDDEN_DIM).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    # Using BCELoss since model outputs Sigmoid probabilities
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_val_acc = 0.0
    best_model_state = None

    print("\n" + "="*60)
    print("Training started...")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 60)

        # Train
        train_loss, train_bit_acc, train_exact = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        # Validate
        val_loss, val_bit_acc, val_exact = validate(
            model, val_loader, criterion, DEVICE
        )

        # Print summary
        print(f"Train - Loss: {train_loss:.4f}, Bit Acc: {train_bit_acc:.4f} ({train_bit_acc*100:.2f}%), Exact: {train_exact:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Bit Acc: {val_bit_acc:.4f} ({val_bit_acc*100:.2f}%), Exact: {val_exact:.4f}")

        # Save best model
        if val_bit_acc > best_val_acc:
            best_val_acc = val_bit_acc
            best_model_state = model.state_dict().copy()
            print(f"✓ New best model! Val Bit Acc: {val_bit_acc:.4f} ({val_bit_acc*100:.2f}%)")

    # Save best model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(best_model_state, 'checkpoints/state_twisting.pth')

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation bit accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Model saved to: checkpoints/state_twisting.pth")
    
    # Check if we hit target
    TARGET_ACC = 0.95  # 95% is close to 100%
    if best_val_acc >= TARGET_ACC:
        print(f"\n✓ SUCCESS! Achieved target accuracy ({TARGET_ACC*100}%)")
    else:
        print(f"\n⚠ Target: {TARGET_ACC*100}%, Achieved: {best_val_acc*100:.2f}%")
        print(f"  Gap: {(TARGET_ACC - best_val_acc)*100:.2f}%")


if __name__ == "__main__":
    main()
