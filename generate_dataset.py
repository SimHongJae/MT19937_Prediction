"""
Generate datasets for MT19937 prediction training (NCC Group approach)
Creates two types of datasets:
1. Tempering dataset: (tempered_output, internal_state) pairs
2. Twisting dataset: (state_triplet, next_state) pairs following MT19937 math
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


def generate_twisting_dataset(num_samples, seed=42):
    """
    Generate dataset for state twisting training (NCC Group approach)
    
    MT19937 state relation:
    MT[i] = f(MT[i-624], MT[i-623], MT[i-227])
    
    :param num_samples: number of samples to generate
    :param seed: random seed
    :return: (X, y) where X is (N, 65) and y is (N, 32)
             X format: [1 bit from MT[i-624], 32 bits from MT[i-623], 32 bits from MT[i-227]]
    """
    print(f"Generating state twisting dataset with {num_samples} samples...")
    
    mt = MT19937(seed=seed)
    
    # Generate enough internal states for MT19937 (624 state buffer + samples)
    # We need 624 states to fill the buffer, then generate samples
    total_needed = 624 + num_samples
    print(f"Generating {total_needed} internal states...")
    
    internal_states = []
    for _ in tqdm(range(total_needed), desc="Internal states"):
        internal, _ = mt.extract_with_internal()
        internal_states.append(int_to_bits(internal))
    
    internal_states = np.array(internal_states, dtype=np.float32)
    
    # Create triplets based on MT19937 state relationship
    print("Creating state triplets...")
    X = []
    y = []
    
    for i in tqdm(range(624, 624 + num_samples), desc="Triplets"):
        # Get the three required states
        # MT[i] depends on MT[i-624], MT[i-623], MT[i-227]
        mt_i_624 = internal_states[i - 624]  # (32,)
        mt_i_623 = internal_states[i - 623]  # (32,)
        mt_i_227 = internal_states[i - 227]  # (32,)
        mt_i = internal_states[i]            # (32,) - target
        
        # Extract only MSB from MT[i-624] (NCC optimization)
        # MSB is the last bit (index 31) in our bit representation
        mt_i_624_msb = mt_i_624[31:32]  # (1,)
        
        # Concatenate: [1 bit, 32 bits, 32 bits] = 65 bits
        triplet = np.concatenate([mt_i_624_msb, mt_i_623, mt_i_227])
        
        X.append(triplet)
        y.append(mt_i)
    
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
    # Configuration (NCC Group used 5M samples)
    # Full production training with 5M samples
    NUM_TEMPERING_SAMPLES = 5000000   # NCC used 5,000,000 for 100% accuracy
    NUM_TWISTING_SAMPLES = 5000000    # NCC used 5,000,000 for 100% accuracy
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

    # Generate twisting dataset
    print("\n" + "="*60)
    print("STEP 2: Generating State Twisting Dataset")
    print("="*60)

    # Use different seed for twisting dataset
    twist_X, twist_y = generate_twisting_dataset(NUM_TWISTING_SAMPLES, seed=SEED+1)

    # Split twisting dataset
    (twist_X_train, twist_y_train,
     twist_X_val, twist_y_val,
     twist_X_test, twist_y_test) = split_dataset(twist_X, twist_y)

    # Save twisting dataset
    print("\nSaving twisting dataset...")
    np.savez_compressed(
        "data/twisting_train.npz",
        X=twist_X_train,
        y=twist_y_train
    )
    np.savez_compressed(
        "data/twisting_val.npz",
        X=twist_X_val,
        y=twist_y_val
    )
    np.savez_compressed(
        "data/twisting_test.npz",
        X=twist_X_test,
        y=twist_y_test
    )

    print(f"Twisting dataset saved:")
    print(f"  Train: {len(twist_X_train)} samples, shape {twist_X_train.shape}")
    print(f"  Val:   {len(twist_X_val)} samples")
    print(f"  Test:  {len(twist_X_test)} samples")

    # Print summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print("\nFiles created in 'data/' directory:")
    print("  tempering_train.npz, tempering_val.npz, tempering_test.npz")
    print("  twisting_train.npz, twisting_val.npz, twisting_test.npz")

    # Print dataset info
    print("\nDataset shapes:")
    print(f"  Tempering X: {temp_X_train.shape} (batch, 32_bits)")
    print(f"  Tempering y: {temp_y_train.shape} (batch, 32_bits)")
    print(f"  Twisting X: {twist_X_train.shape} (batch, 65_bits)")
    print(f"  Twisting y: {twist_y_train.shape} (batch, 32_bits)")

    # Show example
    print("\nExample tempering sample:")
    print(f"  Tempered (input): {temp_X_train[0][:8].astype(int).tolist()}... (first 8 bits)")
    print(f"  Internal (target): {temp_y_train[0][:8].astype(int).tolist()}... (first 8 bits)")

    print("\nExample twisting sample:")
    print(f"  Input shape: {twist_X_train[0].shape} (65 bits: 1 + 32 + 32)")
    print(f"  Target shape: {twist_y_train[0].shape} (32 bits)")
    print(f"  First 5 input bits: {twist_X_train[0][:5].astype(int).tolist()}")


if __name__ == "__main__":
    main()
