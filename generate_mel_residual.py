import torch
import torchaudio
import numpy as np
import pickle
import argparse

from mel_models import MelTransformer, MelLSTM, MelGRU
from preprocess_audio_v2 import mel_tokens_to_audio


def generate_mel_audio_residual(
    checkpoint_path,
    output_path='generated_mel.wav',
    duration_seconds=30,
    temperature=0.9,
    device='cuda',
    top_k=50,
    seed_frames=64,
    use_random_seed=False,
    use_beginning=False,
    oscillation_seconds=0
):
    print("="*60)
    print("Generating I guess")
    print("="*60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model_type = config['model_type']
    n_mels = config['n_mels']
    quantization_levels = config['quantization_levels']
    sample_rate = config['sample_rate']
    hop_length = config['hop_length']
    has_bos = config.get('has_bos', False)
    bos_token_value = config.get('bos_token_value', quantization_levels)

    # Load training data for seeding
    with open('tokens.pkl', 'rb') as f:
        data = pickle.load(f)
    training_mel_tokens = data['mel_tokens']
    training_time_positions = data.get('time_positions', np.arange(len(training_mel_tokens)))

    # Create model
    if model_type == 'transformer':
        model = MelTransformer(
            n_mels=n_mels,
            quantization_levels=quantization_levels
        )
    elif model_type == 'lstm':
        model = MelLSTM(
            n_mels=n_mels,
            quantization_levels=quantization_levels
        )
    else:  # gru
        model = MelGRU(
            n_mels=n_mels,
            quantization_levels=quantization_levels
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # keep in fp32 for generation - fp16 kills logit precision!
    print(f"Model: {model_type.upper()} (fp32 for generation)")
    print(f"Temperature: {temperature}")
    print(f"Duration: {duration_seconds}s")
    print(f"Top-K sampling: {top_k}")
    print(f"Mode: RESIDUAL PREDICTION")
    if has_bos:
        print(f"✓ Time position tracking: ENABLED")
        print(f"  BOS token value: {bos_token_value}")
    if oscillation_seconds > 0:
        print(f"Oscillation: {oscillation_seconds}s (alternating seed/generation)")
    print("-"*60)

    # Seed selection
    if use_beginning:
        # Start from the actual beginning of the piece
        seed_start_idx = 0
        best_seed = training_mel_tokens[seed_start_idx:seed_start_idx + seed_frames]
        seed_time_positions = training_time_positions[seed_start_idx:seed_start_idx + seed_frames]
        best_variance = np.var(best_seed)
        print(f"Using BEGINNING of training data, variance: {best_variance:.1f}")
    elif use_random_seed:
        # Random seed
        seed_start_idx = np.random.randint(0, len(training_mel_tokens) - seed_frames)
        best_seed = training_mel_tokens[seed_start_idx:seed_start_idx + seed_frames]
        seed_time_positions = training_time_positions[seed_start_idx:seed_start_idx + seed_frames]
        best_variance = np.var(best_seed)
        print(f"Random seed from index {seed_start_idx}, variance: {best_variance:.1f}")
    else:
        # Find highest variance seed (original behavior)
        seed_start_idx = 0
        best_seed = training_mel_tokens[:seed_frames]
        seed_time_positions = training_time_positions[:seed_frames]
        best_variance = np.var(best_seed)

        for i in range(0, len(training_mel_tokens) - seed_frames, 1000):
            candidate = training_mel_tokens[i:i + seed_frames]
            var = np.var(candidate)
            if var > best_variance:
                best_variance = var
                best_seed = candidate
                seed_start_idx = i
                seed_time_positions = training_time_positions[i:i + seed_frames]
                if var > 2000:
                    break

        print(f"Best variance seed: {best_variance:.1f}")

    print(f"Seed: {seed_frames} frames, variance: {best_variance:.1f}")
    print(f"Seed time positions: {seed_time_positions[0]} to {seed_time_positions[-1]}")
    print(f"NOTE: First {seed_frames} frames ({seed_frames * hop_length / sample_rate:.2f}s) are REAL audio from training data")
    print(f"      Generation starts after that")

    # Calculate total frames to generate
    frames_per_second = sample_rate / hop_length
    total_frames = int(duration_seconds * frames_per_second)

    print(f"Target: {total_frames} frames ({total_frames * hop_length / sample_rate:.1f}s)")

    # Generation with residual accumulation AND time tracking
    generated_mel = list(best_seed)
    generated_time_positions = list(seed_time_positions)

    # Track current position in piece
    current_time_position = seed_time_positions[-1] + 1

    # Diagnostics
    entropy_values = []

    # Oscillation setup
    oscillation_frames = int(oscillation_seconds * frames_per_second) if oscillation_seconds > 0 else 0
    frames_generated = 0
    current_training_idx = seed_start_idx + seed_frames  # Track position in training data

    print(f"\n{'='*60}")
    print(f"GENERATION STARTS")
    print(f"Starting time position: {current_time_position}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        if model_type in ['lstm', 'gru']:
            # Prime hidden state with time positions
            prime_tensor = torch.tensor([best_seed], dtype=torch.long, device=device)
            prime_time_pos = torch.tensor([seed_time_positions], dtype=torch.long, device=device)
            _, hidden = model(prime_tensor, prime_time_pos, None)

            current_frame = torch.tensor([best_seed[-1]], dtype=torch.long, device=device).unsqueeze(0)
            prev_frame = best_seed[-1]  # Track for residual

            for i in range(total_frames):
                # Check if we should oscillate to training data
                if oscillation_frames > 0 and frames_generated > 0 and frames_generated % oscillation_frames == 0:
                    # Switch to training data for next oscillation_frames
                    print(f"\n  [Oscillation at frame {len(generated_mel)}, time position {current_time_position}]")
                    print(f"  Switching to training data...")

                    # Get next chunk from training data
                    # CRITICAL: Find training data that matches our current time position!
                    # We want continuity, so we look for training data at this time position
                    if current_time_position < len(training_time_positions) - oscillation_frames:
                        # Find the training index that corresponds to our current time position
                        training_idx = np.searchsorted(training_time_positions, current_time_position)
                        training_chunk = training_mel_tokens[training_idx:training_idx + oscillation_frames]
                        training_chunk_time_pos = training_time_positions[training_idx:training_idx + oscillation_frames]

                        print(f"  Using training frames at time positions {training_chunk_time_pos[0]} to {training_chunk_time_pos[-1]}")

                        generated_mel.extend(training_chunk.tolist())
                        generated_time_positions.extend(training_chunk_time_pos.tolist())

                        # Update current time position
                        current_time_position = training_chunk_time_pos[-1] + 1

                        # Reset model state with new training data
                        prime_len = min(seed_frames, len(training_chunk))
                        prime_tensor = torch.tensor([training_chunk[-prime_len:]], dtype=torch.long, device=device)
                        prime_time_pos = torch.tensor([training_chunk_time_pos[-prime_len:]], dtype=torch.long, device=device)
                        _, hidden = model(prime_tensor, prime_time_pos, None)

                        current_frame = torch.tensor([training_chunk[-1]], dtype=torch.long, device=device).unsqueeze(0)
                        prev_frame = training_chunk[-1]

                        frames_generated = 0
                        print(f"  Resumed at time position {current_time_position}\n")
                        continue
                    else:
                        print(f"  WARNING: Reached end of training data, continuing generation")

                # Create time position tensor for current frame
                current_time_pos = torch.tensor([[current_time_position]], dtype=torch.long, device=device)

                # Forward pass WITH TIME POSITION
                output, hidden = model(current_frame, current_time_pos, hidden)

                # output: [1, 1, n_mels, quantization_levels]
                logits = output[0, 0, :, :]  # [n_mels, quantization_levels]

                # Apply temperature
                logits = logits / temperature

                # Sample each mel bin
                next_frame_residual = []
                frame_entropies = []

                for mel_idx in range(n_mels):
                    mel_logits = logits[mel_idx]

                    # Top-k sampling
                    if top_k > 0:
                        top_k_vals, top_k_indices = torch.topk(mel_logits, min(top_k, quantization_levels))
                        mel_logits_filtered = torch.full_like(mel_logits, float('-inf'))
                        mel_logits_filtered[top_k_indices] = top_k_vals
                        mel_logits = mel_logits_filtered

                    probs = torch.softmax(mel_logits.float(), dim=-1)

                    # Diagnostic: track entropy
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    frame_entropies.append(entropy)

                    token = torch.multinomial(probs, 1).item()
                    next_frame_residual.append(token)

                entropy_values.append(np.mean(frame_entropies))

                # Convert residual to absolute value
                # Residual is shifted: actual_residual = token - (quantization_levels // 2)
                next_frame_absolute = []
                for mel_idx in range(n_mels):
                    residual = next_frame_residual[mel_idx] - (quantization_levels // 2)
                    absolute_val = prev_frame[mel_idx] + residual
                    absolute_val = np.clip(absolute_val, 0, quantization_levels - 1)
                    next_frame_absolute.append(int(absolute_val))

                generated_mel.append(next_frame_absolute)
                generated_time_positions.append(current_time_position)

                # Update for next iteration
                current_frame = torch.tensor([[next_frame_absolute]], dtype=torch.long, device=device)
                prev_frame = next_frame_absolute
                current_time_position += 1
                frames_generated += 1

                if (i + 1) % 500 == 0:
                    avg_entropy = np.mean(entropy_values[-500:]) if entropy_values else 0
                    print(f"  Generated {i+1}/{total_frames} frames | Time pos: {current_time_position} | Avg entropy: {avg_entropy:.3f}")

        else:  # Transformer
            context_window = 256
            current = torch.tensor([best_seed], dtype=torch.long, device=device)
            current_time_pos = torch.tensor([seed_time_positions], dtype=torch.long, device=device)

            for i in range(total_frames):
                # Check if we should oscillate to training data
                if oscillation_frames > 0 and frames_generated > 0 and frames_generated % oscillation_frames == 0:
                    print(f"\n  [Oscillation at frame {len(generated_mel)}, time position {current_time_position}]")
                    print(f"  Switching to training data...")

                    # Find training data that matches our current time position
                    if current_time_position < len(training_time_positions) - oscillation_frames:
                        training_idx = np.searchsorted(training_time_positions, current_time_position)
                        training_chunk = training_mel_tokens[training_idx:training_idx + oscillation_frames]
                        training_chunk_time_pos = training_time_positions[training_idx:training_idx + oscillation_frames]

                        print(f"  Using training frames at time positions {training_chunk_time_pos[0]} to {training_chunk_time_pos[-1]}")

                        generated_mel.extend(training_chunk.tolist())
                        generated_time_positions.extend(training_chunk_time_pos.tolist())

                        # Update current time position
                        current_time_position = training_chunk_time_pos[-1] + 1

                        # Reset context with new training data
                        context_len = min(context_window, len(training_chunk))
                        current = torch.tensor([training_chunk[-context_len:]], dtype=torch.long, device=device)
                        current_time_pos = torch.tensor([training_chunk_time_pos[-context_len:]], dtype=torch.long, device=device)

                        frames_generated = 0
                        print(f"  Resumed at time position {current_time_position}\n")
                        continue
                    else:
                        print(f"  WARNING: Reached end of training data, continuing generation")

                # Use last context_window frames
                context = current[:, -context_window:]
                context_time_pos = current_time_pos[:, -context_window:]

                # Forward pass WITH TIME POSITIONS
                output = model(context, context_time_pos)

                # Get prediction for next frame
                logits = output[0, -1, :, :]  # [n_mels, quantization_levels]
                logits = logits / temperature

                # Sample each mel bin
                next_frame_residual = []
                frame_entropies = []

                for mel_idx in range(n_mels):
                    mel_logits = logits[mel_idx]

                    # Top-k sampling
                    if top_k > 0:
                        top_k_vals, top_k_indices = torch.topk(mel_logits, min(top_k, quantization_levels))
                        mel_logits_filtered = torch.full_like(mel_logits, float('-inf'))
                        mel_logits_filtered[top_k_indices] = top_k_vals
                        mel_logits = mel_logits_filtered

                    probs = torch.softmax(mel_logits.float(), dim=-1)

                    # track entropy
                    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                    frame_entropies.append(entropy)

                    token = torch.multinomial(probs, 1).item()
                    next_frame_residual.append(token)

                entropy_values.append(np.mean(frame_entropies))

                # Convert residual to absolute value
                prev_frame = generated_mel[-1]
                next_frame_absolute = []
                for mel_idx in range(n_mels):
                    residual = next_frame_residual[mel_idx] - (quantization_levels // 2)
                    absolute_val = prev_frame[mel_idx] + residual
                    absolute_val = np.clip(absolute_val, 0, quantization_levels - 1)
                    next_frame_absolute.append(int(absolute_val))

                generated_mel.append(next_frame_absolute)
                generated_time_positions.append(current_time_position)

                # Append to context
                next_frame_tensor = torch.tensor([[next_frame_absolute]], dtype=torch.long, device=device)
                next_time_pos_tensor = torch.tensor([[current_time_position]], dtype=torch.long, device=device)
                current = torch.cat([current, next_frame_tensor], dim=1)
                current_time_pos = torch.cat([current_time_pos, next_time_pos_tensor], dim=1)

                current_time_position += 1
                frames_generated += 1

                if (i + 1) % 500 == 0:
                    avg_entropy = np.mean(entropy_values[-500:]) if entropy_values else 0
                    print(f"  Generated {i+1}/{total_frames} frames | Time pos: {current_time_position} | Avg entropy: {avg_entropy:.3f}")

    # Convert to numpy array
    generated_mel = np.array(generated_mel)
    generated_time_positions = np.array(generated_time_positions)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"Generated {len(generated_mel)} frames")
    print(f"Time position range: {generated_time_positions[0]} to {generated_time_positions[-1]}")
    print(f"Shape: {generated_mel.shape}")
    print(f"Min: {generated_mel.min()}, Max: {generated_mel.max()}")
    if entropy_values:
        print(f"Average entropy: {np.mean(entropy_values):.3f}")
        print(f"Entropy std: {np.std(entropy_values):.3f}")

        # Low entropy = collapsed distributions
        if np.mean(entropy_values) < 1.0:
            print("Very low entropy - model distributions might have peaked/collapsed")
    print(f"{'='*60}\n")

    # Convert back to audio using Griffin-Lim
    print("Converting to audio (Griffin-Lim reconstruction)...")
    audio, sr = mel_tokens_to_audio(
        generated_mel,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=hop_length,
        n_mels=n_mels,
        quantization_bits=8
    )

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.95

    # Save
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    torchaudio.save(output_path, audio_tensor, sample_rate)

    print(f"✓ Saved to {output_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate audio from mel-spectrogram model')
    parser.add_argument('checkpoint', help='Model checkpoint path')
    parser.add_argument('--output', default='generated_mel.wav', help='Output audio path')
    parser.add_argument('--random_seed', action='store_true', help='Use random seed instead of best variance')
    parser.add_argument('--beginning', action='store_true', help='Start from beginning of training data (frame 0)')
    parser.add_argument('--oscillation', type=float, default=0, help='Oscillate between generation and training data every N seconds (0 = disabled)')
    parser.add_argument('--duration', type=float, default=30, help='Duration in seconds')
    parser.add_argument('--temperature', type=float, default=0.9, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--seed_frames', type=int, default=64, help='Number of seed frames')
    parser.add_argument('--device', default='cuda', help='Device')

    args = parser.parse_args()

    generate_mel_audio_residual(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        duration_seconds=args.duration,
        temperature=args.temperature,
        device=args.device,
        top_k=args.top_k,
        seed_frames=args.seed_frames,
        use_random_seed=args.random_seed,
        use_beginning=args.beginning,
        oscillation_seconds=args.oscillation
    )
