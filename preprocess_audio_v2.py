import torch
import torchaudio
import numpy as np
from pathlib import Path
import pickle

def process_audio_to_mel_tokens(
    input_dir,
    output_file,
    target_sr=16000,  # Lower SR - better for μ-law
    n_mels=128,       # Mel frequency bins
    n_fft=1024,
    hop_length=256,   # ~16ms per frame at 16kHz
    quantization_bits=8  # 256 levels per mel bin
):
    """
    Convert audio to mel-spectrogram tokens
    This is much better for music than raw waveform μ-law

    Each frame becomes 128 tokens (one per mel bin)
    Better captures musical structure like harmonics, timbre, etc.

    NOW WITH TIME TRACKING:
    - Adds BOS (Beginning Of Sequence) token at start (special value: quantization_levels)
    - Tracks absolute frame position for each frame
    """
    input_path = Path(input_dir)
    audio_files = list(input_path.glob('*.wav')) + \
                  list(input_path.glob('*.mp3')) + \
                  list(input_path.glob('*.flac'))

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files")
    print("=" * 60)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=27.5,  # Lowest piano note (A0)
        f_max=4186,  # Highest piano note (C8)
    )

    all_mel_tokens = []
    total_duration = 0

    for audio_file in sorted(audio_files):
        print(f"Processing {audio_file.name}...")

        # Load audio
        waveform, sr = torchaudio.load(audio_file)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)

        duration = waveform.shape[1] / target_sr
        total_duration += duration

        # Convert to mel spectrogram
        mel_spec = mel_transform(waveform)

        # Convert to log scale (dB) - manual implementation for compatibility
        mel_spec_db = 10.0 * torch.log10(mel_spec + 1e-10)

        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

        # Quantize to n_bits
        mel_tokens = (mel_spec_norm * (2**quantization_bits - 1)).long()

        # Shape: [1, n_mels, time] -> [time, n_mels]
        mel_tokens = mel_tokens.squeeze(0).transpose(0, 1).numpy()

        all_mel_tokens.append(mel_tokens)

        print(f"  Duration: {duration:.1f}s")
        print(f"  Mel frames: {mel_tokens.shape[0]:,}")
        print(f"  Shape: {mel_tokens.shape}")

    # Concatenate all
    all_mel_tokens = np.concatenate(all_mel_tokens, axis=0)

    # ADD BOS TOKEN at the very beginning
    # BOS = special value (quantization_levels = 256) for all mel bins
    quantization_levels = 2**quantization_bits
    bos_frame = np.full((1, n_mels), quantization_levels, dtype=all_mel_tokens.dtype)
    all_mel_tokens = np.concatenate([bos_frame, all_mel_tokens], axis=0)

    # CREATE TIME POSITION ARRAY
    # Frame 0 (BOS) = position 0
    # Frame 1 = position 1, etc.
    time_positions = np.arange(len(all_mel_tokens), dtype=np.int32)

    print("=" * 60)
    print(f"SUMMARY:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  Total duration: {total_duration/60:.1f} minutes")
    print(f"  Total frames: {all_mel_tokens.shape[0]:,} (includes BOS)")
    print(f"  Tokens per frame: {all_mel_tokens.shape[1]}")
    print(f"  Total tokens: {all_mel_tokens.shape[0] * all_mel_tokens.shape[1]:,}")
    print(f"  Sample rate: {target_sr}Hz")
    print(f"  Hop length: {hop_length} ({1000*hop_length/target_sr:.1f}ms)")
    print(f"  Quantization: {quantization_levels} levels (+ 1 BOS token)")
    print(f"  ✓ BOS token added at frame 0 (value={quantization_levels})")
    print(f"  ✓ Time positions tracked: 0 to {len(time_positions)-1}")

    # Save
    data = {
        'mel_tokens': all_mel_tokens,
        'time_positions': time_positions,
        'sample_rate': target_sr,
        'n_mels': n_mels,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'quantization_bits': quantization_bits,
        'quantization_levels': quantization_levels,
        'total_duration': total_duration,
        'num_files': len(audio_files),
        'has_bos': True,  # Flag to indicate this data has BOS token
        'bos_token_value': quantization_levels
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    file_size = Path(output_file).stat().st_size / 1024 / 1024
    print(f"\nSaved to {output_file} ({file_size:.1f} MB)")
    print("=" * 60)

    return all_mel_tokens

def load_mel_token_data(token_file):
    """Load preprocessed mel token data"""
    with open(token_file, 'rb') as f:
        data = pickle.load(f)
    return data

def mel_tokens_to_audio(mel_tokens, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128, quantization_bits=8):
    """
    Convert mel tokens back to audio using inverse mel + Griffin-Lim
    PROPER IMPLEMENTATION with mel-to-linear conversion

    Now handles BOS token properly (skips it)
    """
    quantization_levels = 2**quantization_bits

    # Skip BOS token if present (value would be >= quantization_levels)
    if np.any(mel_tokens[0] >= quantization_levels):
        print("  Skipping BOS token at frame 0")
        mel_tokens = mel_tokens[1:]

    # Dequantize
    mel_spec_norm = mel_tokens.astype(np.float32) / (quantization_levels - 1)

    # Convert back to tensor [time, n_mels] -> [1, n_mels, time]
    mel_spec_norm = torch.tensor(mel_spec_norm).transpose(0, 1).unsqueeze(0)

    # Denormalize (approximate - typical range for piano: -80 to 0 dB)
    mel_spec_db = mel_spec_norm * 80 - 80

    # Convert dB to amplitude (manual implementation for compatibility)
    mel_spec = 10.0 ** (mel_spec_db / 10.0)

    # Create inverse mel filter bank to convert mel -> linear spectrogram
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=27.5,
        f_max=4186.0,
        n_mels=n_mels,
        sample_rate=sample_rate,
        norm=None
    ).T  # Shape: [n_mels, n_freqs]

    # Pseudo-inverse to go from mel to linear
    # mel_spec: [1, n_mels, time]
    # mel_fb: [n_mels, n_freqs]
    # Result: [1, n_freqs, time]
    mel_spec_flat = mel_spec.squeeze(0)  # [n_mels, time]
    linear_spec = torch.matmul(mel_fb.T, mel_spec_flat)  # [n_freqs, time]
    linear_spec = linear_spec.unsqueeze(0)  # [1, n_freqs, time]

    # Now apply Griffin-Lim on the linear spectrogram
    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=32
    )

    audio = griffin_lim(linear_spec)

    return audio.squeeze().numpy(), sample_rate

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess_audio_v2.py <input_dir> [output_file]")
        print("\nExample: python preprocess_audio_v2.py ./recordings mel_tokens.pkl")
        print("\nThis version uses mel-spectrograms instead of raw waveform")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "mel_tokens.pkl"

    process_audio_to_mel_tokens(input_dir, output_file)
