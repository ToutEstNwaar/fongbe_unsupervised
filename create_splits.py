# create_splits.py
import os
import random
import soundfile as sf
from pathlib import Path

# --- Configuration ---
AUDIO_DIR = "fongbe_data/audio_16k"
MANIFEST_DIR = "fongbe_data/manifests"
VALIDATION_HOURS = 3.0  # Target duration for the validation set in hours

# ---------------------

# 1. Get all wav files and their durations
print(f"Scanning {AUDIO_DIR} for .wav files...")
all_files = []
total_duration_seconds = 0
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".wav"):
        file_path = os.path.join(AUDIO_DIR, filename)
        try:
            info = sf.info(file_path)
            duration = info.duration
            frames = info.frames
            all_files.append((filename, frames, duration))
            total_duration_seconds += duration
        except Exception as e:
            print(f"Warning: Could not read {file_path}. Skipping. Error: {e}")

print(f"Found {len(all_files)} audio files, total duration: {total_duration_seconds / 3600:.2f} hours.")

# 2. Shuffle the list randomly
random.seed(42) # Use a fixed seed for reproducibility
random.shuffle(all_files)

# 3. Create validation and train splits
valid_files = []
valid_duration_seconds = 0
train_files = []
train_duration_seconds = 0

target_valid_duration = VALIDATION_HOURS * 3600

for filename, frames, duration in all_files:
    if valid_duration_seconds < target_valid_duration:
        valid_files.append((filename, frames))
        valid_duration_seconds += duration
    else:
        train_files.append((filename, frames))
        train_duration_seconds += duration

print(f"Created train split: {len(train_files)} files, duration: {train_duration_seconds / 3600:.2f} hours.")
print(f"Created valid split: {len(valid_files)} files, duration: {valid_duration_seconds / 3600:.2f} hours.")

# 4. Write the manifest .tsv files
os.makedirs(MANIFEST_DIR, exist_ok=True)
root_path = os.path.abspath(AUDIO_DIR)

with open(os.path.join(MANIFEST_DIR, 'train.tsv'), 'w') as f:
    f.write(root_path + '\n')
    for filename, frames in sorted(train_files):
        f.write(f'{filename}\t{frames}\n')

with open(os.path.join(MANIFEST_DIR, 'valid.tsv'), 'w') as f:
    f.write(root_path + '\n')
    for filename, frames in sorted(valid_files):
        f.write(f'{filename}\t{frames}\n')

print(f"Successfully created train.tsv and valid.tsv in {MANIFEST_DIR}")
