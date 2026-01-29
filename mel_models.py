import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute positional encoding - takes explicit time positions
    """
    def __init__(self, d_model, max_len=500000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, time_positions):
        """
        time_positions: [batch, seq_len] of absolute frame positions
        returns: [batch, seq_len, d_model]
        """
        batch_size, seq_len = time_positions.shape
        # Gather the positional encodings for each position
        pos_encodings = self.pe[time_positions]  # [batch, seq_len, d_model]
        return pos_encodings


class MelTransformer(nn.Module):
    """
    Transformer for mel-spectrogram prediction
    Each timestep predicts 128 mel bins
    """
    def __init__(
        self,
        n_mels=128,
        quantization_levels=256,
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()

        self.n_mels = n_mels
        self.quantization_levels = quantization_levels
        self.d_model = d_model

        # Project mel bins to model dimension
        # Input: [batch, time, n_mels] with values [0, quantization_levels] (includes BOS)
        self.mel_embedding = nn.Embedding(quantization_levels + 1, d_model // n_mels)

        # Combine all mel bins into single vector
        self.input_proj = nn.Linear(d_model, d_model)

        # positional encoding
        self.pos_encoder = AbsolutePositionalEncoding(d_model, max_len=500000)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: predict next mel frame (no BOS in output)
        self.output_proj = nn.Linear(d_model, n_mels * quantization_levels)

    def forward(self, x, time_positions):
        """
        x: [batch, time, n_mels] with values in [0, quantization_levels] (includes BOS)
        time_positions: [batch, time] with absolute frame positions
        returns: [batch, time, n_mels, quantization_levels]
        """
        batch_size, seq_len, n_mels = x.shape

        # Embed each mel bin
        x_embedded = self.mel_embedding(x)  # [batch, time, n_mels, d_model//n_mels]

        # Flatten mel bins
        x_flat = x_embedded.reshape(batch_size, seq_len, self.d_model)

        # Project
        x_flat = self.input_proj(x_flat)

        # Add positional encoding
        pos_encoding = self.pos_encoder(time_positions)
        x_flat = x_flat + pos_encoding

        # Transformer
        x_out = self.transformer(x_flat)

        # Output projection
        logits = self.output_proj(x_out)

        # Reshape to [batch, time, n_mels, quantization_levels]
        logits = logits.reshape(batch_size, seq_len, self.n_mels, self.quantization_levels)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelLSTM(nn.Module):
    """
    LSTM for mel-spectrogram prediction
    """
    def __init__(
        self,
        n_mels=128,
        quantization_levels=256,
        embedding_dim=64,
        hidden_dim=512,
        num_layers=4,
        dropout=0.2
    ):
        super().__init__()

        self.n_mels = n_mels
        self.quantization_levels = quantization_levels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embed each mel bin (includes BOS)
        self.mel_embedding = nn.Embedding(quantization_levels + 1, embedding_dim)

        # Time position embedding - support up to 500k frames (~8 hours at 16kHz with 256 hop)
        self.time_embedding = nn.Embedding(500000, embedding_dim)

        # Project concatenated embeddings (now includes time info)
        self.input_proj = nn.Linear(n_mels * embedding_dim + embedding_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_mels * quantization_levels)

    def forward(self, x, time_positions, hidden=None):
        """
        x: [batch, time, n_mels]
        time_positions: [batch, time]
        returns: [batch, time, n_mels, quantization_levels], hidden
        """
        batch_size, seq_len, n_mels = x.shape

        # Embed each mel bin
        x_embedded = self.mel_embedding(x)  # [batch, time, n_mels, embedding_dim]

        # Flatten mel bins
        x_flat = x_embedded.reshape(batch_size, seq_len, -1)

        # Embed time positions
        time_embedded = self.time_embedding(time_positions)  # [batch, time, embedding_dim]

        # Concatenate mel embeddings with time embedding
        x_with_time = torch.cat([x_flat, time_embedded], dim=-1)

        # Project
        x_proj = self.input_proj(x_with_time)

        # LSTM
        x_out, hidden = self.lstm(x_proj, hidden)

        # Output projection
        logits = self.output_proj(x_out)

        # Reshape
        logits = logits.reshape(batch_size, seq_len, self.n_mels, self.quantization_levels)

        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelGRU(nn.Module):
    def __init__(
        self,
        n_mels=128,
        quantization_levels=256,
        embedding_dim=64,
        hidden_dim=512,
        num_layers=4,
        dropout=0.2
    ):
        super().__init__()

        self.n_mels = n_mels
        self.quantization_levels = quantization_levels

        self.mel_embedding = nn.Embedding(quantization_levels + 1, embedding_dim)
        self.time_embedding = nn.Embedding(500000, embedding_dim)
        self.input_proj = nn.Linear(n_mels * embedding_dim + embedding_dim, hidden_dim)

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_dim, n_mels * quantization_levels)

    def forward(self, x, time_positions, hidden=None):
        batch_size, seq_len, n_mels = x.shape

        x_embedded = self.mel_embedding(x)
        x_flat = x_embedded.reshape(batch_size, seq_len, -1)

        time_embedded = self.time_embedding(time_positions)
        x_with_time = torch.cat([x_flat, time_embedded], dim=-1)

        x_proj = self.input_proj(x_with_time)

        x_out, hidden = self.gru(x_proj, hidden)

        logits = self.output_proj(x_out)
        logits = logits.reshape(batch_size, seq_len, self.n_mels, self.quantization_levels)

        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Mel-Spectrogram Model Comparisons")
    print("=" * 60)

    models = {
        'MelTransformer': MelTransformer(),
        'MelLSTM': MelLSTM(),
        'MelGRU': MelGRU(),
    }

    for name, model in models.items():
        params = model.count_parameters()
        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Size (fp32): {params * 4 / 1024 / 1024:.2f} MB")
        print(f"  Size (fp16): {params * 2 / 1024 / 1024:.2f} MB")

    # Test forward pass
    print("\n" + "=" * 60)
    print("Testing forward pass...")

    batch_size = 4
    seq_len = 128
    n_mels = 128
    x = torch.randint(0, 257, (batch_size, seq_len, n_mels))  # 0-256 (includes BOS)
    time_positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    print(f"Input shape: {x.shape}")
    print(f"Time positions shape: {time_positions.shape}")

    for name, model in models.items():
        if 'Transformer' in name:
            out = model(x, time_positions)
        else:
            out, _ = model(x, time_positions)
        print(f"{name} output shape: {out.shape}")

    print("\n✓ All models workign with time position tracking")
