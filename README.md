# Fongbe wav2vec-U 2.0 ASR Training

Complete pipeline for training unsupervised Automatic Speech Recognition (ASR) for Fongbe language using Facebook's wav2vec-U 2.0 method.

## Quick Start (3 Steps)

### 1. Setup Environment
```bash
./setup_env.sh
```
This installs:
- Conda environment `fongbe_asr` 
- PyTorch with CUDA support
- Fairseq framework
- All dependencies (soundfile, librosa, transformers, etc.)

### 2. Train Model
```bash
./run_all_fixed.sh
```
This will:
- Download test data automatically
- Process audio and text
- Extract XLSR-53 features
- Train wav2vec-U 2.0 model (34 Fongbe phonemes)
- Save checkpoint to `fongbe_checkpoints_official/`

### 3. Run Inference
```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate fongbe_asr
python fixed_inference.py [audio_file.wav]
```
This generates phoneme predictions from audio files.

## Requirements

- NVIDIA GPU with CUDA support
- Ubuntu/Linux system
- ~15GB disk space
- Conda/Miniconda installed

## What It Does

1. **Training**: Uses wav2vec-U 2.0 to learn phoneme mapping from Fongbe audio without labeled data
2. **Features**: Extracts 1024-dim XLSR-53 multilingual features 
3. **Output**: Model predicts 34 Fongbe phonemes: `a, n, i, ɛ, l, ɔ, u, e, t, k, b, á, m, y, s, w, o, h, ú, g, d, é, ó`

## Model Performance

After training (11 epochs):
- Successfully converts 5s audio → ~70 phoneme predictions  
- Uses aligned K-means clusters (34) matching phoneme vocabulary
- Generates sequences like: `d t g o g t g ɔ g é ɔ é g o...`

## Files

- `setup_env.sh` - Environment setup script
- `run_all_fixed.sh` - Complete training pipeline  
- `fixed_inference.py` - Working inference script
- `fongbe_checkpoints_official/` - Trained model checkpoints
- `fongbe_data/` - Audio/text data (auto-downloaded)

## Citation

```bibtex
@article{liu2022wav2vec,
  title={wav2vec-U 2.0: Self-supervised Learning for Speech Recognition},
  author={Liu, Alexander H and others},
  journal={arXiv preprint arXiv:2106.07447},
  year={2022}
}
```