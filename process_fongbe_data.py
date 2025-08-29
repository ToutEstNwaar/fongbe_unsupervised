#!/usr/bin/env python3
"""
Complete Fongbe data processing pipeline for wav2vec-U 2.0
Processes text files and converts them to phonemes using G2P converter
"""

import os
import re
import regex
import sys
from pathlib import Path

def load_g2p_dictionary(dict_path):
    """Load G2P dictionary from file"""
    g2p_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                grapheme, phoneme = line.strip().split('\t', 1)
                g2p_dict[grapheme] = phoneme
    return g2p_dict

def apply_g2p_rules(word, g2p_dict):
    """Apply G2P conversion with contextual nasalization rules"""
    # Handle nasalization rules (from original converter)
    word = re.sub(r'([aeiouɛɔ])n([ptk])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])m([pb])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])n$', r'\1̃', word)
    word = re.sub(r'([aeiouɛɔ])m$', r'\1̃', word)

    # Apply G2P dictionary
    result = []
    i = 0
    while i < len(word):
        found = False
        # Try longest matches first
        for length in range(min(3, len(word) - i), 0, -1):
            substr = word[i:i+length]
            if substr in g2p_dict:
                result.append(g2p_dict[substr])
                i += length
                found = True
                break
        if not found:
            result.append(word[i])
            i += 1

    return ' '.join(result)

def fongbe_wav2vec_normalization(text):
    """Text normalization for wav2vec-U 2.0 compatible with Fongbe"""
    # Modified regex: wav2vec-U but also exclude numbers \p{N}
    # Keeps: Letters (\p{L}), Marks (\p{M}), Apostrophes ('), Spaces, Hyphens (-)
    # Removes: Punctuation, Numbers, Everything else
    filter_r = regex.compile(r"[^\p{L}\p{M}\' \-]")

    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:  # Skip empty lines
            # Apply regex filter (replace non-matching chars with space)
            line = filter_r.sub(" ", line)
            # Convert to lowercase
            line = line.lower()
            # Normalize whitespace
            line = " ".join(line.split())
            if line:  # Only add non-empty lines
                lines.append(line)

    return lines

def text_to_phonemes(text_lines, g2p_dict):
    """Convert text lines to phonemes"""
    phoneme_lines = []
    for line in text_lines:
        words = line.split()
        phoneme_words = []
        for word in words:
            phonemes = apply_g2p_rules(word, g2p_dict)
            phoneme_words.append(phonemes)
        phoneme_lines.append(' '.join(phoneme_words))
    return phoneme_lines

def create_phoneme_dictionary(phoneme_lines):
    """Create phoneme dictionary with frequency counts"""
    phoneme_counts = {}

    for line in phoneme_lines:
        phonemes = line.split()
        for phoneme in phonemes:
            if phoneme:
                phoneme_counts[phoneme] = phoneme_counts.get(phoneme, 0) + 1

    # Sort by frequency (descending)
    sorted_phonemes = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_phonemes

def main():
    if len(sys.argv) != 3:
        print("Usage: python process_fongbe_data.py <data_dir> <g2p_dict_path>")
        print("Example: python process_fongbe_data.py fongbe_data/text codes/dictionary.txt")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    g2p_dict_path = Path(sys.argv[2])

    # Create output directories
    phonemes_dir = data_dir.parent / "phonemes"
    manifests_dir = data_dir.parent / "manifests"
    os.makedirs(phonemes_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    print(f"🔤 Loading G2P dictionary from {g2p_dict_path}...")
    g2p_dict = load_g2p_dictionary(g2p_dict_path)
    print(f"   Loaded {len(g2p_dict)} G2P mappings")

    all_phoneme_lines = []
    processed_files = []

    # Process all text files
    for txt_file in data_dir.glob("*.txt"):
        print(f"📝 Processing {txt_file.name}...")

        with open(txt_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Normalize text
        normalized_lines = fongbe_wav2vec_normalization(raw_text)
        print(f"   Normalized to {len(normalized_lines)} lines")

        # Convert to phonemes
        phoneme_lines = text_to_phonemes(normalized_lines, g2p_dict)
        all_phoneme_lines.extend(phoneme_lines)

        # Save phoneme file
        phoneme_file = phonemes_dir / f"{txt_file.stem}_phonemes.txt"
        with open(phoneme_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(phoneme_lines) + '\n')

        processed_files.append({
            'name': txt_file.stem,
            'phoneme_lines': len(phoneme_lines)
        })

        print(f"   → {phoneme_file} ({len(phoneme_lines)} phoneme lines)")

    # Create phoneme dictionary
    print(f"📚 Creating phoneme dictionary...")
    phoneme_freq = create_phoneme_dictionary(all_phoneme_lines)

    dict_file = manifests_dir / "dict.phn.txt"
    with open(dict_file, 'w', encoding='utf-8') as f:
        for phoneme, count in phoneme_freq:
            f.write(f"{phoneme} {count}\n")

    print(f"   → {dict_file} ({len(phoneme_freq)} unique phonemes)")

    # Create phoneme manifest
    manifest_file = manifests_dir / "train_phonemes.tsv"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        f.write(str(phonemes_dir) + '\n')
        for file_info in processed_files:
            f.write(f"{file_info['name']}_phonemes.txt\t{file_info['phoneme_lines']}\n")

    print(f"   → {manifest_file}")

    print(f"✅ Processing complete!")
    print(f"   📁 Phoneme files: {len(processed_files)}")
    print(f"   🔤 Unique phonemes: {len(phoneme_freq)}")
    print(f"   📝 Total phoneme lines: {len(all_phoneme_lines)}")

if __name__ == "__main__":
    main()
