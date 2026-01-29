# mel_models.py  -- Safeguarded positional handling + small defensive checks

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (relative to sequence position).
    Adds pe to the input (batch, seq_len, d_model).
    """
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # safe slice of pe (device relocation handled by buffer)
        return x + self.pe[:, :x.size(1)]


class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute positional encoding that accepts explicit time positions
    """
    def __init__(self, d_model, max_len=500000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)                    # buffer moves with model

    def forward(self, time_positions):
        """
        time_positions: [batch, seq_len] (long tensor or convertible)
        returns: [batch, seq_len, d_model]
        """
        if not torch.is_tensor(time_positions):
            time_positions = torch.tensor(time_positions, dtype=torch.long)

        # Ensure long dtype and device matches the positional buffer
        time_positions = time_positions.long().to(self.pe.device)

        max_len = self.pe.size(0)
        if (time_positions >= max_len).any():
            # wrap instead of raising to avoid crashes for extremely long datasets
            time_positions = time_positions % max_len

        # Index into the buffer: returns [batch, seq_len, d_model]
        pos_encodings = self.pe[time_positions]
        return pos_encodings


class MelTransformer(nn.Module):
    """
    Transformer for mel-spectrogram prediction.
    Predicts next mel frame (residual tokens) for each of n_mels bins.
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

        # Each mel-bin token value range: [0 .. quantization_levels] (inclusive BOS)
        # embedding size chosen so that n_mels * (d_model // n_mels) == d_model
        emb_dim_per_bin = max(1, d_model // n_mels)
        self.mel_embedding = nn.Embedding(quantization_levels + 1, emb_dim_per_bin)

        # Combine all mel bins into single vector
        self.input_proj = nn.Linear(emb_dim_per_bin * n_mels, d_model)

        # positional encoding (absolute time)
        self.pos_encoder = AbsolutePositionalEncoding(d_model, max_len=500000)

        # Transformer encoder (batch_first=True)
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
        x: [batch, time, n_mels] (LongTensor with token ids, includes BOS potentially)
        time_positions: [batch, time] (LongTensor of absolute frame indices)

        Returns:
          logits: [batch, time, n_mels, quantization_levels]
        """

        batch_size, seq_len, n_mels = x.shape

        # Defensive check: ensure token ids are in valid range for embedding
        max_token_id = int(self.mel_embedding.num_embeddings - 1)
        x_max = int(x.max().item()) if x.numel() > 0 else 0
        if x_max > max_token_id:
            raise ValueError(f"Token id {x_max} out of range (max allowed {max_token_id}). "
                             "Check tokenizer / BOS value.")

        # Embed each mel bin: result [batch, time, n_mels, emb_dim_per_bin]
        x_embedded = self.mel_embedding(x)

        # Flatten mel bins into model dimension
        x_flat = x_embedded.reshape(batch_size, seq_len, -1)  # [batch, time, emb_dim_per_bin*n_mels]
        x_flat = self.input_proj(x_flat)                      # [batch, time, d_model]

        # Safety: ensure time_positions is tensor, long and on same device as model buffers
        if not torch.is_tensor(time_positions):
            time_positions = torch.tensor(time_positions, dtype=torch.long, device=x_flat.device)
        else:
            time_positions = time_positions.long().to(self.pos_encoder.pe.device)

        # Wrap time_positions modulo positional length to avoid OOB indexing
        max_pos_len = self.pos_encoder.pe.size(0)
        if (time_positions >= max_pos_len).any():
            time_positions = time_positions % max_pos_len

        # Get positional encodings (returns [batch, seq_len, d_model])
        pos_encoding = self.pos_encoder(time_positions)

        # Add positional encoding
        x_flat = x_flat + pos_encoding

        # Transformer encoder
        x_out = self.transformer(x_flat)

        # Output projection: [batch, time, n_mels * quantization_levels]
        logits = self.output_proj(x_out)

        # Reshape to [batch, time, n_mels, quantization_levels]
        logits = logits.reshape(batch_size, seq_len, self.n_mels, self.quantization_levels)

        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelLSTM(nn.Module):
    """
    LSTM model for mel-spectrogram prediction (supports time embeddings).
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

        # Time position embedding - supports up to max_entries (default 500k)
        self.time_embedding = nn.Embedding(500000, embedding_dim)

        # Project concatenated embeddings (mel bins flattened + time embedding)
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
        returns: logits [batch, time, n_mels, quantization_levels], hidden
        """
        batch_size, seq_len, n_mels = x.shape

        # Embed mel bins -> [batch, time, n_mels, embedding_dim]
        x_embedded = self.mel_embedding(x)

        # Flatten mel bins
        x_flat = x_embedded.reshape(batch_size, seq_len, -1)

        # Ensure time_positions tensor on same device as time_embedding
        if not torch.is_tensor(time_positions):
            time_positions = torch.tensor(time_positions, dtype=torch.long)
        time_positions = time_positions.long().to(self.time_embedding.weight.device)

        # Wrap time positions modulo embedding size to avoid OOB
        max_pos = self.time_embedding.num_embeddings
        if (time_positions >= max_pos).any():
            time_positions = time_positions % max_pos

        # Embed time positions -> [batch, time, embedding_dim]
        time_embedded = self.time_embedding(time_positions)

        # Concatenate mel embeddings with time embedding
        x_with_time = torch.cat([x_flat, time_embedded], dim=-1)

        # Project and run LSTM
        x_proj = self.input_proj(x_with_time)
        x_out, hidden = self.lstm(x_proj, hidden)

        # Output projection
        logits = self.output_proj(x_out)
        logits = logits.reshape(batch_size, seq_len, self.n_mels, self.quantization_levels)

        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MelGRU(nn.Module):
    """
    GRU model for mel-spectrogram prediction (supports time embeddings).
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

        # Embed mel bins
        x_embedded = self.mel_embedding(x)
        x_flat = x_embedded.reshape(batch_size, seq_len, -1)

        # Ensure time_positions on same device and long dtype
        if not torch.is_tensor(time_positions):
            time_positions = torch.tensor(time_positions, dtype=torch.long)
        time_positions = time_positions.long().to(self.time_embedding.weight.device)

        # Wrap positions modulo available embedding rows
        max_pos = self.time_embedding.num_embeddings
        if (time_positions >= max_pos).any():
            time_positions = time_positions % max_pos

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
    # Quick sanity test (small shapes)
    print("=" * 60)
    print("Mel-Spectrogram Model Comparisons (sanity check)")
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

    # Test forward pass with small random data
    print("\n" + "=" * 60)
    print("Testing forward pass...")

    batch_size = 2
    seq_len = 64
    n_mels = 128
    # tokens in range [0 .. 256] inclusive (256 is BOS)
    x = torch.randint(0, 257, (batch_size, seq_len, n_mels), dtype=torch.long)
    time_positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    print(f"Input shape: {x.shape}")
    print(f"Time positions shape: {time_positions.shape}")

    for name, model in models.items():
        if 'Transformer' in name:
            out = model(x, time_positions)
        else:
            out, _ = model(x, time_positions)
        print(f"{name} output shape: {out.shape}")

    print("\n✓ All models working with safe time position handling")
