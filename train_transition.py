"""
Train State Transition Network
Goal: Learn MT19937 twisting/state transition function
Input: Sequence of 624 internal states (624, 32)
Output: Next internal state (32 bits)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os

from model import StateTransition


def load_transition_data(split='train'):
    """Load transition dataset"""
    data = np.load(f'data/transition_{split}.npz')
    X = torch.from_numpy(data['X'])  # (N, 624, 32) sequences
    y = torch.from_numpy(data['y'])  # (N, 32) next state
    return X, y


def calculate_bit_accuracy(predictions, targets):
    """Calculate bit-wise accuracy"""
    pred_bits = (torch.sigmoid(predictions) > 0.5).float()
    correct = (pred_bits == targets).float().mean()
    return correct.item()


def calculate_exact_match(predictions, targets):
    """Calculate exact match rate (all 32 bits correct)"""
    pred_bits = (torch.sigmoid(predictions) > 0.5).float()
    all_match = (pred_bits == targets).all(dim=1).float().mean()
    return all_match.item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_bit_acc = 0.0
    total_exact = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(X_batch)

        # Calculate loss
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
            'bit_acc': f'{bit_acc*100:.1f}%',
            'exact': f'{exact*100:.1f}%'
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
        for X_batch, y_batch in tqdm(dataloader, desc="Validation"):
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
    # Hyperparameters (optimized for GTX 1060 3GB and REDUCED window size)
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1

    BATCH_SIZE = 64  # Can use larger batch with smaller window (100 vs 624)
    LEARNING_RATE = 1e-4  # Slightly higher LR for faster convergence
    EPOCHS = 50
    WARMUP_EPOCHS = 5

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("STATE TRANSITION NETWORK TRAINING")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: Transformer")
    print(f"  d_model: {D_MODEL}")
    print(f"  nhead: {NHEAD}")
    print(f"  num_layers: {NUM_LAYERS}")
    print(f"  dim_feedforward: {DIM_FEEDFORWARD}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    X_train, y_train = load_transition_data('train')
    X_val, y_val = load_transition_data('val')

    print(f"Train: {len(X_train)} samples, shape {X_train.shape}")
    print(f"Val: {len(X_val)} samples, shape {X_val.shape}")

    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = StateTransition(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            return 0.5 ** ((epoch - WARMUP_EPOCHS) / 20)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Training loop
    best_val_bit_acc = 0.0
    best_model_state = None

    print("\n" + "="*60)
    print("Training started...")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Train
        train_loss, train_bit_acc, train_exact = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS
        )

        # Validate
        print("")
        val_loss, val_bit_acc, val_exact = validate(
            model, val_loader, criterion, DEVICE
        )

        # Update learning rate
        scheduler.step()

        # Print summary
        print(f"\n{'Results':^60}")
        print(f"{'-'*60}")
        print(f"{'Split':<10} {'Loss':<10} {'Bit Acc':<15} {'Exact Match':<15}")
        print(f"{'-'*60}")
        print(f"{'Train':<10} {train_loss:<10.4f} {train_bit_acc*100:<14.2f}% {train_exact*100:<14.2f}%")
        print(f"{'Val':<10} {val_loss:<10.4f} {val_bit_acc*100:<14.2f}% {val_exact*100:<14.2f}%")
        print(f"{'-'*60}")

        # Save best model
        if val_bit_acc > best_val_bit_acc:
            best_val_bit_acc = val_bit_acc
            best_model_state = model.state_dict().copy()
            print(f"\n✓ New best model! Val Bit Acc: {val_bit_acc*100:.2f}%")

            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(best_model_state, 'checkpoints/state_transition.pth')
            print(f"  Saved to: checkpoints/state_transition.pth")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation bit accuracy: {best_val_bit_acc*100:.2f}%")
    print(f"Model saved to: checkpoints/state_transition.pth")

    # Compare with target
    target_acc = 0.80
    if best_val_bit_acc >= target_acc:
        print(f"\n✓ SUCCESS: Achieved target accuracy of {target_acc*100:.0f}%!")
    else:
        print(f"\n⚠ Target accuracy: {target_acc*100:.0f}%, Achieved: {best_val_bit_acc*100:.2f}%")
        print(f"  Gap: {(target_acc - best_val_bit_acc)*100:.2f}%")


if __name__ == "__main__":
    main()
