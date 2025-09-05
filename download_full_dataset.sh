#!/bin/bash

# Download the complete Fongbe dataset from HuggingFace
# Downloads all audio and text files from the vnwaar/fongbe dataset

set -e  # Exit on error

echo "🎯 Downloading complete Fongbe dataset from HuggingFace..."
echo "📖 Dataset: vnwaar/fongbe"
echo

# Create data directory
DATA_DIR="fongbe_data"
echo "📁 Creating directory: $DATA_DIR"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Method 1: Try using huggingface-cli (recommended)
download_with_cli() {
    echo "🔄 Attempting to download using huggingface-cli..."
    
    if command -v huggingface-cli &> /dev/null; then
        echo "✅ huggingface-cli found"
        huggingface-cli download --repo-type dataset vnwaar/fongbe --local-dir . --local-dir-use-symlinks False
        return 0
    else
        echo "❌ huggingface-cli not found"
        return 1
    fi
}

# Method 2: Try using git clone
download_with_git() {
    echo "🔄 Attempting to download using git clone..."
    
    if command -v git &> /dev/null; then
        echo "✅ git found"
        
        # Check if git lfs is available
        if ! command -v git-lfs &> /dev/null; then
            echo "⚠️  git-lfs not found, large files might not download properly"
        else
            git lfs install
        fi
        
        git clone https://huggingface.co/datasets/vnwaar/fongbe .
        return 0
    else
        echo "❌ git not found"
        return 1
    fi
}

# Method 3: Direct download using wget/curl
download_with_wget() {
    echo "🔄 Attempting to download using direct file download..."
    
    BASE_URL="https://huggingface.co/datasets/vnwaar/fongbe/resolve/main"
    
    # Create subdirectories
    mkdir -p audio text
    
    # List of biblical books (based on the dataset structure we found)
    BOOKS=(
        "1CH" "1CO" "1JN" "1KI" "1PE" "1SA" "1TH" "1TI"
        "2CH" "2CO" "2JN" "2KI" "2PE" "2SA" "2TH" "2TI"
        "3JN" "ACT" "AMO" "COL" "DAN" "DEU" "ECC" "EPH"
        "EST" "EXO" "EZE" "EZR" "GAL" "GEN" "HAB" "HAG"
        "HEB" "HOS" "ISA" "JAS" "JDG" "JER" "JOB" "JOL"
        "JON" "JOS" "JUD" "LAM" "LEV" "LUK" "MAL" "MAR"
        "MAT" "MIC" "NAH" "NEH" "NUM" "OBA" "PHI" "PHM"
        "PRO" "PSA" "REV" "ROM" "RUT" "SNG" "TIT" "ZEC"
        "ZEP"
    )
    
    total_files=0
    downloaded_files=0
    
    echo "📊 Attempting to download files for ${#BOOKS[@]} books..."
    
    for book in "${BOOKS[@]}"; do
        echo "📖 Processing book: $book"
        
        # Try to download up to 50 chapters per book (most books have fewer)
        for chapter in {1..50}; do
            audio_file="${book}.${chapter}.FON13.wav"
            text_file="${book}.${chapter}.FON13.txt"
            
            audio_url="$BASE_URL/$audio_file"
            text_url="$BASE_URL/$text_file"
            
            # Try to download audio file
            if wget -q --spider "$audio_url" 2>/dev/null; then
                echo "📥 Downloading $audio_file..."
                if wget -q "$audio_url" -O "audio/$audio_file"; then
                    ((downloaded_files++))
                    echo "   ✅ Downloaded audio/$audio_file"
                else
                    echo "   ❌ Failed to download $audio_file"
                fi
                ((total_files++))
            else
                # If audio file doesn't exist, assume no more chapters for this book
                break
            fi
            
            # Try to download text file
            if wget -q --spider "$text_url" 2>/dev/null; then
                echo "📥 Downloading $text_file..."
                if wget -q "$text_url" -O "text/$text_file"; then
                    ((downloaded_files++))
                    echo "   ✅ Downloaded text/$text_file"
                else
                    echo "   ❌ Failed to download $text_file"
                fi
                ((total_files++))
            fi
        done
        echo
    done
    
    echo "📊 Download summary:"
    echo "   Total files attempted: $total_files"
    echo "   Successfully downloaded: $downloaded_files"
    
    if [ $downloaded_files -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Try download methods in order of preference
SUCCESS=false

if download_with_cli; then
    echo "✅ Successfully downloaded using huggingface-cli!"
    SUCCESS=true
elif download_with_git; then
    echo "✅ Successfully downloaded using git clone!"
    SUCCESS=true
elif download_with_wget; then
    echo "✅ Successfully downloaded using direct download!"
    SUCCESS=true
else
    echo "❌ All download methods failed!"
fi

if [ "$SUCCESS" = true ]; then
    # Organize files into audio/text subdirectories
    echo
    echo "🗂️ Organizing files into audio/text subdirectories..."
    mkdir -p audio text
    
    # Move .wav files if they exist in the current directory
    if ls *.wav 1> /dev/null 2>&1; then
        echo "   Moving .wav files to audio/..."
        mv *.wav audio/
    else
        echo "   No .wav files found in the root directory to move."
    fi
    
    # Move .txt files if they exist in the current directory
    if ls *.txt 1> /dev/null 2>&1; then
        echo "   Moving .txt files to text/..."
        mv *.txt text/
    else
        echo "   No .txt files found in the root directory to move."
    fi
    
    echo "✅ File organization complete."

    # Analyze downloaded data
    echo
    echo "📊 Analyzing downloaded dataset..."
    
    wav_count=$(find . -name "*.wav" -type f | wc -l)
    txt_count=$(find . -name "*.txt" -type f | wc -l)
    other_count=$(find . -type f ! -name "*.wav" ! -name "*.txt" | wc -l)
    
    echo "🎵 Audio files (.wav): $wav_count"
    echo "📝 Text files (.txt): $txt_count"
    echo "📄 Other files: $other_count"
    
    # Calculate total size
    total_size=$(du -sh . | cut -f1)
    echo "💾 Total size: $total_size"
    
    # Show some example files
    echo
    echo "📋 Example files:"
    find . -name "*.wav" -type f | head -5 | while read -r file; do
        echo "   $file"
    done
    if [ $wav_count -gt 5 ]; then
        echo "   ... and $((wav_count - 5)) more audio files"
    fi
    
    echo
    echo "🎉 Dataset download completed successfully!"
    echo "📁 Dataset location: $(pwd)"
    echo "🚀 Ready to use the complete Fongbe dataset!"
else
    echo
    echo "❌ Dataset download failed!"
    echo "💡 Possible solutions:"
    echo "   1. Install huggingface-cli: pip install huggingface_hub"
    echo "   2. Install git and git-lfs"
    echo "   3. Check your internet connection"
    echo "   4. Try running with authentication: huggingface-cli login"
    exit 1
fi