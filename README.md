# LMbutAudio
A neural network in Python for playing back music via generation; to get perfect at one song (in this case Fantaisie Impromptu as an example)

## Prerequisites
- Linux, or WSL. I'm sure this would work on Windows but I'm unsure of how to 'transplant' it.
- Nvidia GPU with cuda
- Python 3.x (3.13 used here)

## Linux Setup

Create a new folder `mkdir ~/LMbutAudio`

cd into the directory `cd ~/LMbutAudio`

Create the venv `python -m venv .venv && source .venv/bin/activate`

Install dependencies: `pip install tqdm torch torchaudio numpy torchcodec`

Example file structure after downloading and pasting from [releases](https://github.com/Pook-gif/LMbutAudio/releases/):

```
(.venv) Gentoo@Gentoo ~/Downloads/whattheaids $ tree
.
├── output
├── recordings
│   ├── Fantaisie Impromptu 1.wav
│   ├── Fantaisie Impromptu 10.wav
│   ├── Fantaisie Impromptu 11.wav
│   ├── Fantaisie Impromptu 2.wav
│   ├── Fantaisie Impromptu 3.wav
│   ├── Fantaisie Impromptu 4.wav
│   ├── Fantaisie Impromptu 5.wav
│   ├── Fantaisie Impromptu 6.wav
│   ├── Fantaisie Impromptu 8.wav
│   ├── Fantaisie Impromptu 9.wav
│   └── Fantaisie Impromptu 7.wav
├── generate_mel_residual.py
├── mel_models.py
├── preprocess_audio_v2.py
└── train_mel.py 
```

## Usage
0. Obtain your audio files and ensure consistency. Aim for higher than 30 minutes of sample size, feel free to experiment with several different songs instead of just one (like how it's supposed to be).
1. Preprocess the audio `python preprocess_audio_v2.py /home/Gentoo/Downloads/whattheaids/recordings tokens.pkl` - Should create a new file called 'tokens.pkl' and have outputted something along:
```
SUMMARY:
  Total files: 11
  Total duration: 59.3 minutes
  Total frames: 222,210 (includes BOS)
  Tokens per frame: 128
  Total tokens: 28,442,880
  Sample rate: 16000Hz
  Hop length: 256 (16.0ms)
  Quantization: 256 levels (+ 1 BOS token)
  ✓ BOS token added at frame 0 (value=256)
  ✓ Time positions tracked: 0 to 222209

Saved to tokens.pkl (217.9 MB)
```
2. Train a neural network `python train_mel.py --model_type transformer --epochs 100 --batch_size 20 --seq_length 256 --save_dir output/` - This is a minimal example, experiment according to your hardware constraints but this is what works well for me on 8GB of VRAM without spilling to system RAM. By default you will have the latest iteration of the model come out every 10 epoch unless you use the `--checkpoint-every` flag.
- Try out python train_mel.py --help to see what flags can also be used (e.g: frequency of writing to disk by epoch, model type, device, learning rate, etc)
3. Generate `python generate_mel_residual.py /home/Gentoo/Downloads/whattheaids/output/checkpoint_epoch_10.pt --duration 30 --temperature 0.7 --output /home/Gentoo/Downloads/whattheaids/output/generated.wav`
- Just like training, make sure to check the flags with the --help flag. Output should look alike to this when complete:
```
GENERATION COMPLETE
Generated 1939 frames
Time position range: 201000 to 202938
Shape: (1939, 128)
Min: 0, Max: 255
Average entropy: 2.846
Entropy std: 0.021
```
### Done!
The final .wav should be.. somewhere in there unless you redirected it. Experiment with oscillation, seq length, batch size, temperature, more training data, etc, for better results

## Flags
### train_mel.py
-  --token_file TOKEN_FILE   |   Path to mel token file
-  --checkpoint-every CHECKPOINT_EVERY   |   Save a checkpoint every N epochs (0 = disable)
-  --model_type {transformer,lstm,gru}   |   Neural Network type
-  --seq_length SEQ_LENGTH   |   Sequence length in frames
-  --batch_size BATCH_SIZE   |   Batch size
-  --epochs EPOCHS | Number of epochs
-  --lr LR | Learning rate
-  --warmup_epochs WARMUP_EPOCHS | Warmup epochs
-  --no_amp | Disable AMP
-  --gradient_clip GRADIENT_CLIP | Gradient clipping
-  --device DEVICE | Device (e.g: cpu, cuda)
-  --save_dir SAVE_DIR | Checkpoint/model directory
  
### generate_mel_residual.py
-  --output OUTPUT | Output audio path
-  --random_seed | Use random seed instead of best variance
-  --beginning | Start from beginning of training data (frame 0)
-  --oscillation OSCILLATION | Oscillate between generation and training data every N seconds (0 = disabled). Very useful on low VRAM GPUs
-  --duration DURATION | Duration in seconds
-  --temperature TEMPERATURE | Sampling temperature
-  --top_k TOP_K | Top-k sampling
-  --seed_frames SEED_FRAMES | Number of seed frames
-  --device DEVICE | Device (e.g: cpu, cuda)
##
