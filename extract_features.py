#!/usr/bin/env python3
"""
Extract wav2vec 2.0 features from audio files for wav2vec-U 2.0 training
Supports XLSR-53 multilingual model and handles audio resampling
"""

import torch
import soundfile as sf
import librosa
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import fairseq

def resample_audio_file(input_path, output_path, target_sr=16000):
    """Resample audio file to target sample rate and convert to mono"""
    print(f"🎵 Processing: {input_path}")

    # Load audio file
    wav, sr = sf.read(input_path)
    print(f"   Original: {sr}Hz, shape: {wav.shape}")

    # Convert to mono if stereo
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)  # Average channels
        print(f"   Converted to mono: {wav.shape}")

    # Resample to 16kHz
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        print(f"   Resampled to {target_sr}Hz: {wav.shape}")

    # Save resampled audio
    sf.write(output_path, wav, target_sr)
    print(f"   → Saved to: {output_path}")

    return len(wav)

def create_audio_manifest(audio_dir, output_manifest):
    """Create fairseq-compatible audio manifest"""
    manifest_lines = [str(audio_dir)]

    for audio_file in sorted(audio_dir.glob("*.wav")):
        # Get audio length in samples
        wav, sr = sf.read(audio_file)
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)

        manifest_lines.append(f"{audio_file.name}\t{len(wav)}")

    with open(output_manifest, 'w') as f:
        f.write('\n'.join(manifest_lines) + '\n')

    return len(manifest_lines) - 1  # Subtract header line

def extract_wav2vec_features(audio_manifest, model_path, output_dir, layer=14):
    """Extract wav2vec 2.0 features using fairseq"""

    print(f"🤖 Loading wav2vec 2.0 model from {model_path}...")

    # Set up fairseq extraction command
    cmd = [
        "python", "fairseq/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py",
        str(audio_manifest.parent),
        "--split", "train",
        "--save-dir", str(output_dir),
        "--checkpoint", str(model_path),
        "--layer", str(layer)
    ]

    print(f"🚀 Extracting features (layer {layer})...")
    print(f"   Command: {' '.join(cmd)}")

    # Run extraction
    import subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/fairseq:" + env.get("PYTHONPATH", "")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Feature extraction failed:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

    print(f"✅ Feature extraction complete!")
    return True

def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_features.py <audio_dir> <model_path> <output_dir>")
        print("Example: python extract_features.py fongbe_data/audio xlsr_53_56k.pt fongbe_features")
        sys.exit(1)

    audio_dir = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])

    if not audio_dir.exists():
        print(f"❌ Audio directory not found: {audio_dir}")
        sys.exit(1)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Create directories
    resampled_dir = audio_dir.parent / "audio_16khz"
    manifests_dir = audio_dir.parent / "manifests"
    os.makedirs(resampled_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"🎵 Resampling audio files to 16kHz...")

    # Resample all audio files to 16kHz mono
    audio_files = list(audio_dir.glob("*.wav"))
    if not audio_files:
        print(f"❌ No .wav files found in {audio_dir}")
        sys.exit(1)

    for audio_file in audio_files:
        output_file = resampled_dir / audio_file.name
        resample_audio_file(audio_file, output_file)

    # Create audio manifest
    print(f"📄 Creating audio manifest...")
    manifest_path = manifests_dir / "train.tsv"
    num_files = create_audio_manifest(resampled_dir, manifest_path)
    print(f"   → {manifest_path} ({num_files} files)")

    # Extract features
    success = extract_wav2vec_features(manifest_path, model_path, output_dir)

    if success:
        # Copy dictionary to features directory for training
        dict_source = manifests_dir / "dict.phn.txt"
        dict_target = output_dir / "dict.phn.txt"
        if dict_source.exists():
            import shutil
            shutil.copy(dict_source, dict_target)
            print(f"📚 Copied dictionary to {dict_target}")

        print(f"✅ All done!")
        print(f"   🎵 Audio files: {num_files}")
        print(f"   📁 Features saved to: {output_dir}")
        print(f"   🔤 Ready for wav2vec-U 2.0 training!")
    else:
        print(f"❌ Feature extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
