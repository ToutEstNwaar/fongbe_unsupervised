# Fongbe wav2vec-U 2.0 ASR Training

A complete pipeline for training unsupervised Automatic Speech Recognition (ASR) for Fongbe language using Facebook's wav2vec-U 2.0 method with XLSR-53 multilingual features.

## Overview

This project implements the wav2vec-U 2.0 unsupervised ASR training pipeline specifically adapted for Fongbe, a West African language. It uses:

- **XLSR-53**: Multilingual wav2vec 2.0 model trained on 56k hours across 53 languages
- **wav2vec-U 2.0**: Unsupervised ASR training method from Facebook Research
- **Custom G2P**: Grapheme-to-Phoneme conversion tailored for Fongbe phonology
- **Tone-aware processing**: Preserves Fongbe tone marks throughout the pipeline

## Quick Start

### 1. Environment Setup (one-time)

```bash
# Set up the complete environment
./setup_environment.sh
source /opt/miniforge3/bin/activate fongbe_asr
```

### 2. Run Complete Pipeline (includes test data download)

```bash
# Complete pipeline: downloads test data → processing → training
./run_fongbe_asr_training.sh
```

**That's it!** 🎯 The script will automatically:
- Download 3 Fongbe audio/text pairs for testing
- Process text → phonemes
- Extract XLSR-53 features from audio
- Train wav2vec-U 2.0 model

### Alternative: Download Test Data Separately

```bash
# If you want to download test data first
python download_test_data.py
```

## Manual Steps

If you prefer to run steps individually:

### Step 1: Process Text Data
```bash
python process_fongbe_data.py fongbe_data/text codes/dictionary.txt
```

### Step 2: Extract Features
```bash
python extract_features.py fongbe_data/audio xlsr_53_56k.pt fongbe_features
```

### Step 3: Train Model
```bash
python train_fongbe_wav2vec_u2.py fongbe_features
```

## Data Format

### Audio Requirements
- Format: 16-bit WAV files
- Any sample rate (automatically resampled to 16kHz)
- Mono or stereo (automatically converted to mono)

### Text Requirements
- UTF-8 encoded text files
- Fongbe orthography with tone marks
- One transcript per audio file

### G2P Dictionary Format
```
# Grapheme-to-phoneme mappings (tab-separated)
a       a
kp      kp
gb      gb
```

## Features

- **Multilingual Foundation**: Uses XLSR-53 model trained on 53 languages
- **Tone Preservation**: Maintains Fongbe tone marks (á, è, ɔ̃, etc.)
- **Automatic Audio Processing**: Handles resampling and format conversion
- **Memory Efficient**: Optimized for GPU memory constraints
- **Complete Pipeline**: End-to-end automation from raw data to trained model

## Architecture

```
Raw Audio + Text
       ↓
[Text Normalization] → Phoneme sequences
       ↓
[Audio Resampling] → 16kHz mono WAV
       ↓
[XLSR-53 Features] → 1024-dim features
       ↓
[wav2vec-U 2.0] → Unsupervised ASR model
```

## Model Architecture

- **Generator**: Conv1d layers with batch normalization
- **Discriminator**: 2-layer CNN with GELU activation
- **Segmenter**: Joint segmentation strategy
- **Features**: 1024-dim XLSR-53 representations
- **Targets**: 64-dim pseudo-label space

## Output

Training produces:
- **Model checkpoints**: `/workspace/fongbe_checkpoints/`
- **Training logs**: `/workspace/fongbe_tensorboard/`
- **Features**: `/workspace/fongbe_features/`

Monitor training with:
```bash
tensorboard --logdir /workspace/fongbe_tensorboard
```

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- Conda/Miniconda
- ~15GB disk space (for models and features)

## Citation

If you use this code, please cite:

```bibtex
@article{liu2022wav2vec,
  title={wav2vec-U 2.0: Self-supervised Pre-training for Universal Speech Representation Learning},
  author={Liu, Alexander H and Lee, Wei-Ning and others},
  journal={arXiv preprint arXiv:2106.07447},
  year={2022}
}
```

## License

Based on fairseq (MIT License). See individual file headers for specific licensing information.
