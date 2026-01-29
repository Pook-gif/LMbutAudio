import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math

from mel_models import MelTransformer, MelLSTM, MelGRU
from preprocess_audio_v2 import load_mel_token_data


class MelDataset(Dataset):
    """Dataset for mel-spectrogram sequences with time position tracking"""
    def __init__(self, mel_tokens, time_positions, seq_length=128, stride=None):
        """
        mel_tokens: [time, n_mels] array (includes BOS at frame 0)
        time_positions: [time] array of absolute frame positions
        Each sample is a window of 'seq_length' frames
        """
        self.mel_tokens = mel_tokens
        self.time_positions = time_positions
        self.seq_length = seq_length
        self.stride = stride if stride else seq_length // 2

    def __len__(self):
        return max(1, (len(self.mel_tokens) - self.seq_length) // self.stride)

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        start_idx = min(start_idx, len(self.mel_tokens) - self.seq_length - 1)

        # Input: frames [start:start+seq_length]
        # Target: frames [start+1:start+seq_length+1]
        x = torch.tensor(self.mel_tokens[start_idx:start_idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.mel_tokens[start_idx+1:start_idx+self.seq_length+1], dtype=torch.long)

        # Time positions for input frames
        time_pos = torch.tensor(self.time_positions[start_idx:start_idx+self.seq_length], dtype=torch.long)

        return x, y, time_pos


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Learning rate schedule with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_mel_model(
    token_file='tokens.pkl',
    model_type='transformer',  # 'transformer', 'lstm', or 'gru'
    seq_length=128,
    batch_size=16,
    epochs=100,
    learning_rate=0.0003,
    warmup_epochs=5,
    device='cuda',
    save_dir='checkpoints_mel',
    use_amp=True,
    gradient_clip=1.0,
    checkpoint_every=10
):

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load data
    print("=" * 60)
    print("Loading mel-spectrogram data...")
    data = load_mel_token_data(token_file)
    mel_tokens = data['mel_tokens']
    time_positions = data.get('time_positions', np.arange(len(mel_tokens)))
    n_mels = data['n_mels']
    quantization_levels = data['quantization_levels']
    sample_rate = data['sample_rate']
    hop_length = data['hop_length']
    has_bos = data.get('has_bos', False)
    bos_token_value = data.get('bos_token_value', quantization_levels)

    print(f"Total frames: {len(mel_tokens):,}")
    print(f"Duration: {len(mel_tokens)*hop_length/sample_rate/60:.1f} minutes")
    print(f"Mel bins: {n_mels}")
    print(f"Quantization levels: {quantization_levels}")
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Hop length: {hop_length} ({1000*hop_length/sample_rate:.1f}ms per frame)")
    if has_bos:
        print(f"✓ BOS token present at frame 0 (value={bos_token_value})")
        print(f"✓ Time position tracking enabled")

    # Create dataset
    dataset = MelDataset(mel_tokens, time_positions, seq_length=seq_length, stride=seq_length//2)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if device == 'cuda' else False
    )

    print(f"Dataset: {len(dataset):,} sequences (seq_len={seq_length}, stride={seq_length//2})")
    print(f"Batches per epoch: {len(dataloader)}")
    print("=" * 60)

    # Create model
    if model_type == 'transformer':
        model = MelTransformer(
            n_mels=n_mels,
            quantization_levels=quantization_levels,
            d_model=256,
            nhead=8,
            num_layers=8,
            dim_feedforward=1024
        )
    elif model_type == 'lstm':
        model = MelLSTM(
            n_mels=n_mels,
            quantization_levels=quantization_levels,
            embedding_dim=64,
            hidden_dim=512,
            num_layers=4
        )
    else:  # gru
        model = MelGRU(
            n_mels=n_mels,
            quantization_levels=quantization_levels,
            embedding_dim=64,
            hidden_dim=512,
            num_layers=4
        )

    model = model.to(device)

    params = model.count_parameters()
    print(f"Model: {model_type.upper()}")
    print(f"Parameters: {params:,}")
    print(f"Model size: {params * 4 / 1024 / 1024:.2f} MB (fp32)")
    if use_amp:
        print(f"Model size: {params * 2 / 1024 / 1024:.2f} MB (fp16 with AMP)")
    print("=" * 60)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Scheduler
    num_training_steps = len(dataloader) * epochs
    num_warmup_steps = len(dataloader) * warmup_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # AMP scaler
    scaler = GradScaler(enabled=use_amp)

    # Training loop
    print("Starting training...")
    if use_amp:
        print("✓ AMP enabled (fp16 training)")
    print(f"✓ Learning rate warmup: {warmup_epochs} epochs")
    print(f"✓ Gradient clipping: {gradient_clip}")
    print(f"✓ Time position tracking: ENABLED")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y, time_pos) in enumerate(pbar):
            x, y, time_pos = x.to(device), y.to(device), time_pos.to(device)

            optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(enabled=use_amp):
                if model_type in ['lstm', 'gru']:
                    output, _ = model(x, time_pos)
                else:
                    output = model(x, time_pos)

                # output: [batch, time, n_mels, quantization_levels]
                # y: [batch, time, n_mels]

                # Compute residuals
                y_residual = y.clone()
                for b in range(y.shape[0]):
                    for t in range(y.shape[1]):
                        if t == 0:
                            # First frame: use difference from last frame of x
                            y_residual[b, t] = y[b, t] - x[b, -1]
                        else:
                            # Subsequent frames: use difference from previous frame
                            y_residual[b, t] = y[b, t] - y[b, t-1]

                # Shift residuals to be in valid range [0, quantization_levels-1]
                y_residual = (y_residual + quantization_levels // 2).clamp(0, quantization_levels - 1)

                # Compute loss for each mel bin
                batch_size, seq_len, n_mels, n_classes = output.shape
                ce_loss = criterion(
                    output.reshape(batch_size * seq_len * n_mels, n_classes),
                    y_residual.reshape(-1)
                )

                probs = torch.softmax(output.reshape(batch_size * seq_len * n_mels, n_classes), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

                # cross-entropy - small entropy bonus
                loss = ce_loss - 0.03 * entropy

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(output, dim=-1)
                acc = (predicted == y_residual).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })

        # Epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        perplexity = math.exp(min(avg_loss, 20))

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f} | Perplexity: {perplexity:.2f} | LR: {current_lr:.6f}")

        # Periodic checkpoints
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            current_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_acc,
                'perplexity': perplexity,
                'config': {
                    'model_type': model_type,
                    'n_mels': n_mels,
                    'quantization_levels': quantization_levels,
                    'sample_rate': sample_rate,
                    'hop_length': hop_length,
                    'seq_length': seq_length,
                    'has_bos': has_bos,
                    'bos_token_value': bos_token_value
                }
            }
            torch.save(current_checkpoint, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")

        print("-" * 60)

    # Training complete - OUTSIDE the epoch loop
    print("=" * 60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best model saved to: {save_dir / 'best_model.pt'}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train mel-spectrogram music generation model')
    parser.add_argument('--token_file', type=str, default='tokens.pkl', help='Path to mel token file')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Save a checkpoint every N epochs (0 = disable')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'lstm', 'gru'])
    parser.add_argument('--seq_length', type=int, default=128, help='Sequence length in frames')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--save_dir', type=str, default='checkpoints_mel', help='Checkpoint directory')

    args = parser.parse_args()

    train_mel_model(
        token_file=args.token_file,
        model_type=args.model_type,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        save_dir=args.save_dir,
        use_amp=not args.no_amp,
        gradient_clip=args.gradient_clip,
        checkpoint_every=args.checkpoint_every
    )
