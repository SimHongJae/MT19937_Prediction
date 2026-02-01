"""
Neural network models for MT19937 prediction
Implements three models:
1. InverseTempering: Reverses MT19937 tempering function
2. StateTransition: Predicts next internal state from sequence
3. MT19937Predictor: Combined end-to-end predictor
"""

import torch
import torch.nn as nn
import math


class InverseTempering(nn.Module):
    """
    MLP to reverse MT19937 tempering function (NCC Group architecture)
    Input: 32 bits after tempering
    Output: 32 bits before tempering (internal state)
    
    Architecture from NCC Group research:
    - Single hidden layer with 640 neurons
    - ReLU activation for hidden layer
    - Sigmoid activation for output layer
    - Total params: 41,632 (vs previous 8,352)
    - Achieved: 100% bit accuracy
    """

    def __init__(self, hidden_dim=640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, hidden_dim),  # 32*640 + 640 = 21,120 params
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),  # 640*32 + 32 = 20,512 params
            nn.Sigmoid()  # Binary output for bits
        )

    def forward(self, tempered_bits):
        """
        :param tempered_bits: (batch, 32) float tensor of bits [0.0 or 1.0]
        :return: (batch, 32) predicted internal state bits (probabilities 0-1)
        """
        return self.net(tempered_bits)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer
    """

    def __init__(self, d_model, max_len=1000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: (batch, seq_len, d_model)
        :return: x + positional encoding
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class StateTwisting(nn.Module):
    """
    MLP to learn MT19937 state twisting function (NCC Group architecture)
    
    MT19937 state relation:
    MT[i] = f(MT[i-624], MT[i-623], MT[i-227])
    
    Input optimization from NCC Group:
    - MT[i-624]: Only MSB (1 bit) - other bits masked to 0x80000000
    - MT[i-623]: All 32 bits
    - MT[i-227]: All 32 bits
    - Total: 65 bits instead of 96
    
    Architecture from NCC Group research:
    - Single hidden layer with 96 neurons
    - ReLU activation for hidden layer
    - Sigmoid activation for output layer
    - Total params: 9,440
    - Achieved: 100% bit accuracy
    """

    def __init__(self, hidden_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, hidden_dim),   # 65*96 + 96 = 6,336 params
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),   # 96*32 + 32 = 3,104 params
            nn.Sigmoid()  # Binary output for bits
        )

    def forward(self, state_triplet):
        """
        :param state_triplet: (batch, 65) float tensor
                              [1 bit from MT[i-624], 32 bits from MT[i-623], 32 bits from MT[i-227]]
        :return: (batch, 32) predicted next state bits (probabilities 0-1)
        """
        return self.net(state_triplet)


class MT19937Predictor(nn.Module):
    """
    Combined model: Inverse Tempering + State Twisting (NCC Group approach)
    Input: Three tempered outputs at positions [i-624, i-623, i-227]
    Output: Next tempered output at position i
    """

    def __init__(self,
                 inverse_temp_hidden=640,
                 twisting_hidden=96):
        super().__init__()

        # Module 1: Inverse tempering
        self.inverse_tempering = InverseTempering(hidden_dim=inverse_temp_hidden)

        # Module 2: State twisting
        self.state_twisting = StateTwisting(hidden_dim=twisting_hidden)

    def forward(self, tempered_triplet):
        """
        :param tempered_triplet: (batch, 3, 32) three tempered outputs
                                 [MT[i-624], MT[i-623], MT[i-227]]
        :return: (batch, 32) predicted next internal state (probabilities 0-1)
        """
        batch_size = tempered_triplet.shape[0]

        # Reverse tempering for all three outputs
        # Reshape to (batch * 3, 32)
        tempered_flat = tempered_triplet.view(-1, 32)

        # Apply inverse tempering
        internal_flat = self.inverse_tempering(tempered_flat)  # (batch * 3, 32)

        # Reshape back to (batch, 3, 32)
        internal_states = internal_flat.view(batch_size, 3, 32)

        # Extract bits according to NCC optimization:
        # - MT[i-624]: MSB only (1 bit)
        # - MT[i-623]: All 32 bits
        # - MT[i-227]: All 32 bits
        mt_624_msb = internal_states[:, 0:1, 31:32]  # (batch, 1, 1) - MSB
        mt_623_all = internal_states[:, 1:2, :]      # (batch, 1, 32)
        mt_227_all = internal_states[:, 2:3, :]      # (batch, 1, 32)

        # Concatenate: (batch, 65)
        state_triplet = torch.cat([
            mt_624_msb.squeeze(1),   # (batch, 1)
            mt_623_all.squeeze(1),   # (batch, 32)
            mt_227_all.squeeze(1)    # (batch, 32)
        ], dim=1)

        # Predict next state using twisting model
        next_state_probs = self.state_twisting(state_triplet)

        return next_state_probs

    def load_pretrained_modules(self, inverse_temp_path=None, state_trans_path=None):
        """
        Load pre-trained weights for individual modules
        :param inverse_temp_path: path to inverse tempering weights
        :param state_trans_path: path to state transition weights
        """
        if inverse_temp_path:
            state_dict = torch.load(inverse_temp_path)
            self.inverse_tempering.load_state_dict(state_dict)
            print(f"Loaded inverse tempering weights from {inverse_temp_path}")

        if state_trans_path:
            state_dict = torch.load(state_trans_path)
            self.state_transition.load_state_dict(state_dict)
            print(f"Loaded state transition weights from {state_trans_path}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")

    # Test InverseTempering
    print("="*60)
    print("InverseTempering Model")
    print("="*60)
    inv_temp = InverseTempering(hidden_dim=640)
    test_input = torch.randn(4, 32)  # batch=4, bits=32
    test_output = inv_temp(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Parameters: {count_parameters(inv_temp):,}")

    # Test StateTwisting
    print("\n" + "="*60)
    print("StateTwisting Model (NCC Group Architecture)")
    print("="*60)
    state_twist = StateTwisting(hidden_dim=96)
    test_input = torch.randn(4, 65)  # batch=4, 65 bits (1 + 32 + 32)
    test_output = state_twist(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Parameters: {count_parameters(state_twist):,}")

    # Test MT19937Predictor
    print("\n" + "="*60)
    print("MT19937Predictor (Combined) Model")
    print("="*60)
    predictor = MT19937Predictor(
        inverse_temp_hidden=640,
        twisting_hidden=96
    )
    test_input = torch.randn(4, 3, 32)  # batch=4, 3 states, bits=32
    test_output = predictor(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Parameters: {count_parameters(predictor):,}")

    print("\n" + "="*60)
    print("All models initialized successfully!")
    print("="*60)
