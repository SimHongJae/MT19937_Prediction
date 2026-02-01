"""
Test MT19937 Prediction Models
Evaluates trained models on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

from model import InverseTempering, StateTransition, MT19937Predictor


def load_data(data_type, split='test'):
    """Load dataset"""
    data = np.load(f'data/{data_type}_{split}.npz')
    X = torch.from_numpy(data['X'])
    y = torch.from_numpy(data['y'])
    return X, y


def calculate_bit_accuracy(predictions, targets):
    """Calculate bit-wise accuracy"""
    pred_bits = (torch.sigmoid(predictions) > 0.5).float()
    correct = (pred_bits == targets).float().mean()
    return correct.item()


def calculate_exact_match(predictions, targets):
    """Calculate exact match rate"""
    pred_bits = (torch.sigmoid(predictions) > 0.5).float()
    all_match = (pred_bits == targets).all(dim=1).float().mean()
    return all_match.item()


def calculate_per_bit_accuracy(predictions, targets):
    """Calculate accuracy for each of the 32 bits"""
    pred_bits = (torch.sigmoid(predictions) > 0.5).float()
    per_bit_acc = (pred_bits == targets).float().mean(dim=0)
    return per_bit_acc.cpu().numpy()


def test_model(model, dataloader, device, model_name="Model"):
    """Test a model on dataset"""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_bit_acc = 0.0
    total_exact = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    print(f"\nEvaluating {model_name}...")
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Testing"):
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

            all_predictions.append(predictions.cpu())
            all_targets.append(y_batch.cpu())

    # Aggregate all predictions
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Calculate metrics
    avg_loss = total_loss / num_batches
    avg_bit_acc = total_bit_acc / num_batches
    avg_exact = total_exact / num_batches

    per_bit_acc = calculate_per_bit_accuracy(all_predictions, all_targets)

    return {
        'loss': avg_loss,
        'bit_accuracy': avg_bit_acc,
        'exact_match': avg_exact,
        'per_bit_accuracy': per_bit_acc
    }


def print_results(results, model_name="Model"):
    """Print test results"""
    print("\n" + "="*60)
    print(f"{model_name} TEST RESULTS")
    print("="*60)
    print(f"Loss:            {results['loss']:.4f}")
    print(f"Bit Accuracy:    {results['bit_accuracy']:.4f} ({results['bit_accuracy']*100:.2f}%)")
    print(f"Exact Match:     {results['exact_match']:.4f} ({results['exact_match']*100:.2f}%)")
    print("="*60)

    # Per-bit accuracy analysis
    per_bit = results['per_bit_accuracy']
    print(f"\nPer-bit accuracy statistics:")
    print(f"  Mean: {per_bit.mean():.4f}")
    print(f"  Std:  {per_bit.std():.4f}")
    print(f"  Min:  {per_bit.min():.4f} (bit {per_bit.argmin()})")
    print(f"  Max:  {per_bit.max():.4f} (bit {per_bit.argmax()})")

    # Show first and last 8 bits
    print(f"\nFirst 8 bits accuracy: {per_bit[:8]}")
    print(f"Last 8 bits accuracy:  {per_bit[-8:]}")


def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 64

    print("="*60)
    print("MT19937 MODEL EVALUATION")
    print("="*60)
    print(f"Device: {DEVICE}")

    # Test Inverse Tempering Model
    print("\n" + "="*60)
    print("1. INVERSE TEMPERING MODEL")
    print("="*60)

    # Load test data
    X_test, y_test = load_data('tempering', 'test')
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test samples: {len(X_test)}")

    # Load model
    inv_temp_model = InverseTempering(hidden_dim=64).to(DEVICE)
    try:
        state_dict = torch.load('checkpoints/inverse_tempering.pth', map_location=DEVICE)
        inv_temp_model.load_state_dict(state_dict)
        print("OK Loaded model from checkpoints/inverse_tempering.pth")

        # Test
        results = test_model(inv_temp_model, test_loader, DEVICE, "Inverse Tempering")
        print_results(results, "Inverse Tempering")

    except FileNotFoundError:
        print("X Model checkpoint not found: checkpoints/inverse_tempering.pth")
        print("  Please train the model first using train_inverse_tempering.py")

    # Test State Transition Model
    print("\n" + "="*60)
    print("2. STATE TRANSITION MODEL")
    print("="*60)

    # Load test data
    X_test, y_test = load_data('transition', 'test')
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test samples: {len(X_test)}, shape: {X_test.shape}")

    # Load model
    state_trans_model = StateTransition(
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(DEVICE)

    try:
        state_dict = torch.load('checkpoints/state_transition.pth', map_location=DEVICE)
        state_trans_model.load_state_dict(state_dict)
        print("OK Loaded model from checkpoints/state_transition.pth")

        # Test
        results = test_model(state_trans_model, test_loader, DEVICE, "State Transition")
        print_results(results, "State Transition")

        # Check if target accuracy achieved
        target = 0.80
        if results['bit_accuracy'] >= target:
            print(f"\nOK SUCCESS: Achieved target accuracy of {target*100:.0f}%!")
        else:
            print(f"\n⚠ Target: {target*100:.0f}%, Achieved: {results['bit_accuracy']*100:.2f}%")

    except FileNotFoundError:
        print("X Model checkpoint not found: checkpoints/state_transition.pth")
        print("  Please train the model first using train_transition.py")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
