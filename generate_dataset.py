"""
Generate datasets for MT19937 prediction training
Creates two types of datasets:
1. Tempering dataset: (tempered_output, internal_state) pairs
2. Transition dataset: (624 internal states, next internal state) sequences
"""

import numpy as np
from mt19937 import MT19937, int_to_bits
from tqdm import tqdm
import os


def generate_tempering_dataset(num_samples, seed=42):
    """
    Generate dataset for inverse tempering training
    :param num_samples: number of samples to generate
    :param seed: random seed
    :return: (tempered_bits, internal_bits) arrays of shape (N, 32)
    """
    print(f"Generating tempering dataset with {num_samples} samples...")

    mt = MT19937(seed=seed)

    tempered_bits = []
    internal_bits = []

    for _ in tqdm(range(num_samples), desc="Tempering data"):
        internal, tempered = mt.extract_with_internal()

        # Convert to bit arrays
        internal_bits.append(int_to_bits(internal))
        tempered_bits.append(int_to_bits(tempered))

    return np.array(tempered_bits, dtype=np.float32), np.array(internal_bits, dtype=np.float32)


def generate_transition_dataset(num_samples, window_size=100, seed=42):
    """
    Generate dataset for state transition training
    :param num_samples: number of samples to generate
    :param window_size: sequence length (default 624)
    :param seed: random seed
    :return: (X, y) where X is (N, 624, 32) and y is (N, 32)
    """
    print(f"Generating transition dataset with {num_samples} samples...")

    mt = MT19937(seed=seed)

    # Generate enough internal states
    total_needed = num_samples + window_size
    print(f"Generating {total_needed} internal states...")

    internal_states = []
    for _ in tqdm(range(total_needed), desc="Internal states"):
        internal, _ = mt.extract_with_internal()
        internal_states.append(int_to_bits(internal))

    internal_states = np.array(internal_states, dtype=np.float32)

    # Create sliding windows
    print("Creating sliding windows...")
    X = []
    y = []

    for i in tqdm(range(num_samples), desc="Windows"):
        window = internal_states[i:i+window_size]  # (624, 32)
        target = internal_states[i+window_size]    # (32,)

        X.append(window)
        y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/val/test
    :param X: input data
    :param y: target data
    :param train_ratio: fraction for training
    :param val_ratio: fraction for validation
    :param test_ratio: fraction for testing
    :return: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]

    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    # Configuration
    NUM_TEMPERING_SAMPLES = 100000  # For inverse tempering (smaller, easier task)
    NUM_TRANSITION_SAMPLES = 100000  # Reduced from 500000 due to memory constraints
    WINDOW_SIZE = 100  # Reduced from 624 to make problem more learnable
    SEED = 42

    # Create output directory
    os.makedirs("data", exist_ok=True)

    # Generate tempering dataset
    print("\n" + "="*60)
    print("STEP 1: Generating Tempering Dataset")
    print("="*60)

    tempered_X, internal_y = generate_tempering_dataset(NUM_TEMPERING_SAMPLES, seed=SEED)

    # Split tempering dataset
    (temp_X_train, temp_y_train,
     temp_X_val, temp_y_val,
     temp_X_test, temp_y_test) = split_dataset(tempered_X, internal_y)

    # Save tempering dataset
    print("\nSaving tempering dataset...")
    np.savez_compressed(
        "data/tempering_train.npz",
        X=temp_X_train,
        y=temp_y_train
    )
    np.savez_compressed(
        "data/tempering_val.npz",
        X=temp_X_val,
        y=temp_y_val
    )
    np.savez_compressed(
        "data/tempering_test.npz",
        X=temp_X_test,
        y=temp_y_test
    )

    print(f"Tempering dataset saved:")
    print(f"  Train: {len(temp_X_train)} samples")
    print(f"  Val:   {len(temp_X_val)} samples")
    print(f"  Test:  {len(temp_X_test)} samples")

    # Generate transition dataset
    print("\n" + "="*60)
    print("STEP 2: Generating Transition Dataset")
    print("="*60)

    # Use different seed for transition dataset
    trans_X, trans_y = generate_transition_dataset(NUM_TRANSITION_SAMPLES, window_size=WINDOW_SIZE, seed=SEED+1)

    # Split transition dataset
    (trans_X_train, trans_y_train,
     trans_X_val, trans_y_val,
     trans_X_test, trans_y_test) = split_dataset(trans_X, trans_y)

    # Save transition dataset
    print("\nSaving transition dataset...")
    np.savez_compressed(
        "data/transition_train.npz",
        X=trans_X_train,
        y=trans_y_train
    )
    np.savez_compressed(
        "data/transition_val.npz",
        X=trans_X_val,
        y=trans_y_val
    )
    np.savez_compressed(
        "data/transition_test.npz",
        X=trans_X_test,
        y=trans_y_test
    )

    print(f"Transition dataset saved:")
    print(f"  Train: {len(trans_X_train)} samples, shape {trans_X_train.shape}")
    print(f"  Val:   {len(trans_X_val)} samples")
    print(f"  Test:  {len(trans_X_test)} samples")

    # Print summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print("\nFiles created in 'data/' directory:")
    print("  tempering_train.npz, tempering_val.npz, tempering_test.npz")
    print("  transition_train.npz, transition_val.npz, transition_test.npz")

    # Print dataset info
    print("\nDataset shapes:")
    print(f"  Tempering X: {temp_X_train.shape} (batch, 32_bits)")
    print(f"  Tempering y: {temp_y_train.shape} (batch, 32_bits)")
    print(f"  Transition X: {trans_X_train.shape} (batch, {WINDOW_SIZE}_states, 32_bits)")
    print(f"  Transition y: {trans_y_train.shape} (batch, 32_bits)")

    # Show example
    print("\nExample tempering sample:")
    print(f"  Tempered (input): {temp_X_train[0][:8].astype(int).tolist()}... (first 8 bits)")
    print(f"  Internal (target): {temp_y_train[0][:8].astype(int).tolist()}... (first 8 bits)")

    print("\nExample transition sample:")
    print(f"  Input shape: {trans_X_train[0].shape} (624 states, each 32 bits)")
    print(f"  Target shape: {trans_y_train[0].shape} (32 bits)")


if __name__ == "__main__":
    main()
