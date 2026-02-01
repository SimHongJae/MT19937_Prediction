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
    MLP to reverse MT19937 tempering function
    Input: 32 bits after tempering
    Output: 32 bits before tempering (internal state)
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)
        )

    def forward(self, tempered_bits):
        """
        :param tempered_bits: (batch, 32) float tensor of bits [0.0 or 1.0]
        :return: (batch, 32) predicted internal state bits (logits)
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


class StateTransition(nn.Module):
    """
    Transformer-based model to predict next internal state
    Input: Sequence of 624 internal states (624, 32)
    Output: Next internal state (32 bits)
    """

    def __init__(self, d_model=128, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Input embedding: project 32-bit vectors to d_model dimensions
        self.input_embedding = nn.Linear(32, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=624)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Linear(d_model, 32)

    def forward(self, internal_states):
        """
        :param internal_states: (batch, 624, 32) float tensor of internal state bits
        :return: (batch, 32) predicted next state bits (logits)
        """
        # Embed: (batch, 624, 32) -> (batch, 624, d_model)
        x = self.input_embedding(internal_states)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer: (batch, 624, d_model) -> (batch, 624, d_model)
        x = self.transformer(x)

        # Take last position (or use pooling)
        # For MT19937, the last position should be most relevant for predicting next
        x = x[:, -1, :]  # (batch, d_model)

        # Output: (batch, d_model) -> (batch, 32)
        return self.output_head(x)


class MT19937Predictor(nn.Module):
    """
    Combined model: Inverse Tempering + State Transition
    Input: Sequence of 624 tempered outputs
    Output: Next tempered output
    """

    def __init__(self,
                 inverse_temp_hidden=64,
                 trans_d_model=128,
                 trans_nhead=4,
                 trans_num_layers=4,
                 trans_dim_feedforward=512,
                 trans_dropout=0.1):
        super().__init__()

        # Module 1: Inverse tempering
        self.inverse_tempering = InverseTempering(hidden_dim=inverse_temp_hidden)

        # Module 2: State transition
        self.state_transition = StateTransition(
            d_model=trans_d_model,
            nhead=trans_nhead,
            num_layers=trans_num_layers,
            dim_feedforward=trans_dim_feedforward,
            dropout=trans_dropout
        )

        # Note: We could add a tempering module here, but since we're predicting
        # the next internal state, we'll apply tempering in post-processing if needed

    def forward(self, tempered_sequence):
        """
        :param tempered_sequence: (batch, 624, 32) sequence of tempered outputs
        :return: (batch, 32) predicted next internal state (logits)
        """
        batch_size, seq_len, bit_dim = tempered_sequence.shape

        # Reverse tempering for each element in sequence
        # Reshape to (batch * seq_len, 32)
        tempered_flat = tempered_sequence.view(-1, bit_dim)

        # Apply inverse tempering
        internal_flat = self.inverse_tempering(tempered_flat)  # (batch * seq_len, 32)

        # Apply sigmoid to get probabilities, then round to get bits
        # (Or keep as logits for training)
        internal_probs = torch.sigmoid(internal_flat)

        # Reshape back to (batch, seq_len, 32)
        internal_sequence = internal_probs.view(batch_size, seq_len, bit_dim)

        # Predict next state using transition model
        next_state_logits = self.state_transition(internal_sequence)

        return next_state_logits

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
    inv_temp = InverseTempering(hidden_dim=64)
    test_input = torch.randn(4, 32)  # batch=4, bits=32
    test_output = inv_temp(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Parameters: {count_parameters(inv_temp):,}")

    # Test StateTransition
    print("\n" + "="*60)
    print("StateTransition Model")
    print("="*60)
    state_trans = StateTransition(d_model=128, nhead=4, num_layers=4)
    test_input = torch.randn(4, 624, 32)  # batch=4, seq=624, bits=32
    test_output = state_trans(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Parameters: {count_parameters(state_trans):,}")

    # Test MT19937Predictor
    print("\n" + "="*60)
    print("MT19937Predictor (Combined) Model")
    print("="*60)
    predictor = MT19937Predictor(
        inverse_temp_hidden=64,
        trans_d_model=128,
        trans_nhead=4,
        trans_num_layers=4
    )
    test_input = torch.randn(4, 624, 32)  # batch=4, seq=624, bits=32
    test_output = predictor(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    print(f"Parameters: {count_parameters(predictor):,}")

    print("\n" + "="*60)
    print("All models initialized successfully!")
    print("="*60)
