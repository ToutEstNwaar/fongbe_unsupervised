#!/bin/bash

# A robust script to download the complete Fongbe dataset from HuggingFace.
# This script uses 'git clone' with Git LFS, which is more reliable for
# repositories with thousands of files than 'huggingface-cli'.
# It is also resumable in case of network interruptions.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
DATASET_REPO="https://huggingface.co/datasets/vnwaar/fongbe"
DATA_DIR="fongbe_data"

# --- Main Script Logic ---

echo "🎯 Downloading complete Fongbe dataset from HuggingFace..."
echo "📖 Dataset: vnwaar/fongbe"
echo "⚙️ Method: git clone with LFS (most reliable)"
echo

# 1. Check for required tools
echo "🔎 Checking for required tools (git and git-lfs)..."
if ! command -v git &> /dev/null; then
    echo "❌ Error: git is not installed. Please install it to continue."
    exit 1
fi
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Error: git-lfs is not installed. Please install it to continue."
    exit 1
fi
echo "✅ All required tools are found."
echo

# 2. Handle download: clone if new, or pull to resume if it exists
# Check if the directory exists and is a valid git repository
if [ -d "$DATA_DIR/.git" ]; then
    echo "📁 Found existing dataset directory: $DATA_DIR"
    echo "🔄 Attempting to resume download..."
    cd "$DATA_DIR"
    
    echo "   1. Cleaning up any previous partial state (git reset --hard)..."
    git reset --hard
    
    echo "   2. Downloading missing large files (git lfs pull)..."
    git lfs pull

else
    echo "📁 Creating new directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    echo "🚀 Cloning dataset repository... (This will take a while)"
    # The initial 'git clone' will also download all LFS files.
    git clone "$DATASET_REPO" .
fi

echo
echo "✅ Download process complete."

# 3. Organize files into audio/text subdirectories
echo
echo "🗂️ Organizing files into audio/text subdirectories..."
mkdir -p audio text

# Use 'find' to robustly move files, avoiding errors if no files match
if find . -maxdepth 1 -name "*.wav" -print -quit | grep -q .; then
    echo "   Moving .wav files to audio/..."
    mv *.wav audio/
else
    echo "   No .wav files to move; likely already organized."
fi

if find . -maxdepth 1 -name "*.txt" -print -quit | grep -q .; then
    echo "   Moving .txt files to text/..."
    mv *.txt text/
else
    echo "   No .txt files to move; likely already organized."
fi
echo "✅ File organization complete."

# 4. Verify that all text files have a corresponding audio file
echo
echo "🔍 Verifying integrity: Checking for missing audio files..."
missing_count=0
for txt_file in text/*.txt; do
    # Construct the expected wav file path
    wav_file="audio/$(basename "${txt_file%.txt}.wav")"
    if [ ! -f "$wav_file" ]; then
        echo "   ⚠️ Missing corresponding audio for: $(basename "$txt_file")"
        ((missing_count++))
    fi
done

if [ "$missing_count" -eq 0 ]; then
    echo "✅ Verification successful! All audio files are present."
else
    echo "❌ Verification FAILED! Found $missing_count missing audio files."
    echo "   The download may be incomplete. Please try running the script again."
    exit 1
fi

# 5. Final analysis of the downloaded dataset
echo
echo "📊 Analyzing downloaded dataset..."

wav_count=$(find audio/ -type f -name "*.wav" | wc -l)
txt_count=$(find text/ -type f -name "*.txt" | wc -l)

echo "🎵 Audio files (.wav): $wav_count"
echo "📝 Text files (.txt): $txt_count"

# Calculate total size
total_size=$(du -sh . | cut -f1)
echo "💾 Total size: $total_size"
echo

echo "🎉 Dataset download and setup completed successfully!"
echo "📁 Dataset location: $(pwd)"
echo "🚀 Ready to use the complete Fongbe dataset!"
