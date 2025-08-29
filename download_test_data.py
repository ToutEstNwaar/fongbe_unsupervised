#!/usr/bin/env python3
"""
Download Fongbe test data from HuggingFace dataset
Downloads 3 audio files and their corresponding transcripts for testing
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm

def download_file(url, local_path):
    """Download a file with progress bar"""
    print(f"📥 Downloading {local_path.name}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(local_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

    print(f"   ✅ Saved to {local_path}")

def main():
    # HuggingFace dataset base URL
    base_url = "https://huggingface.co/datasets/vnwaar/fongbe/resolve/main"

    # Test files to download
    test_files = [
        ("1CH.1.FON13.wav", "1CH.1.FON13.txt"),
        ("1CH.10.FON13.wav", "1CH.10.FON13.txt"),
        ("1CH.11.FON13.wav", "1CH.11.FON13.txt")
    ]

    # Create directories
    data_dir = Path("fongbe_data")
    audio_dir = data_dir / "audio"
    text_dir = data_dir / "text"

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    print("🎯 Downloading Fongbe test data from HuggingFace...")
    print(f"📁 Audio files → {audio_dir}")
    print(f"📁 Text files → {text_dir}")
    print()

    total_files = len(test_files) * 2
    completed = 0

    for audio_file, text_file in test_files:
        try:
            # Download audio file
            audio_url = f"{base_url}/{audio_file}"
            audio_path = audio_dir / audio_file
            download_file(audio_url, audio_path)
            completed += 1

            # Download text file
            text_url = f"{base_url}/{text_file}"
            text_path = text_dir / text_file
            download_file(text_url, text_path)
            completed += 1

            print(f"   📊 Progress: {completed}/{total_files} files complete")
            print()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading {audio_file}/{text_file}: {e}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

    if completed == total_files:
        print("✅ All test data downloaded successfully!")
        print()
        print("📋 Downloaded files:")
        print(f"   🎵 Audio: {len(test_files)} files in {audio_dir}")
        print(f"   📝 Text: {len(test_files)} files in {text_dir}")
        print()
        print("🚀 Ready to run training pipeline!")
        print("   Next step: ./run_fongbe_asr_training.sh")
    else:
        print(f"⚠️  Downloaded {completed}/{total_files} files")
        print("Some files may have failed to download.")

if __name__ == "__main__":
    main()
