#!/bin/bash
set -ex

# Combined and corrected training script with absolute paths

# Configuration
DATA_DIR="fongbe_data"
G2P_DICT="codes/dictionary.txt"
MODEL_PATH="xlsr_53_56k.pt"
FEATURES_DIR="fongbe_features_official"
TEXT_FEATURES_DIR="fongbe_text_features"
LANGUAGE="fongbe"
CLUSTERS=64
LAYER=14
MIN_PHONES=1
SIL_PROB=0.5
MIN_WORD_COUNT=1

WORKSPACE_DIR="/workspace/fongbe_unsupervised"

echo "🎯 Unified Fongbe wav2vec-U 2.0 ASR Training Pipeline"
echo "====================================================="

# Activate conda environment
echo "🔧 Activating conda environment 'fongbe_asr'..."
if ! source /opt/miniforge3/bin/activate fongbe_asr; then
    echo "❌ Failed to activate conda environment 'fongbe_asr'"
    echo "Please run setup_env.sh first to create the environment"
    exit 1
fi

# Verify environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "fongbe_asr" ]]; then
    echo "❌ Environment activation failed. Current env: $CONDA_DEFAULT_ENV"
    echo "Expected: fongbe_asr"
    exit 1
fi

echo "✅ Environment 'fongbe_asr' activated successfully"

# Set required environment variables
export FAIRSEQ_ROOT="$WORKSPACE_DIR/fairseq"
export PYTHONPATH=$FAIRSEQ_ROOT

cd $WORKSPACE_DIR

# Step 1: Data download and preparation
if [ ! -d "$DATA_DIR/audio" ] || [ ! -d "$DATA_DIR/text" ] || [ -z "$(ls -A $DATA_DIR/audio 2>/dev/null)" ]; then
    echo "📥 Test data not found. Downloading from HuggingFace..."
    python3 download_test_data.py
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Downloading XLSR-53 model..."
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt
fi

# Step 2: Prepare audio data and create manifests
echo "📂 Step 2: Converting audio to 16kHz mono and creating manifests..."
mkdir -p "$DATA_DIR/manifests"
mkdir -p "$DATA_DIR/audio_16k"

if [ ! -f "$DATA_DIR/audio_16k/.converted" ]; then
    echo "🔄 Converting audio files to 16kHz mono..."
    python3 -c "
import os, sys, soundfile as sf, librosa
from pathlib import Path
audio_dir = '$DATA_DIR/audio'
output_dir = '$DATA_DIR/audio_16k'
os.makedirs(output_dir, exist_ok=True)
for f in os.listdir(audio_dir):
    if f.endswith('.wav'):
        input_path = os.path.join(audio_dir, f)
        output_path = os.path.join(output_dir, f)
        if not os.path.exists(output_path):
            audio, sr = librosa.load(input_path, sr=16000, mono=True)
            sf.write(output_path, audio, 16000)
Path(os.path.join(output_dir, '.converted')).touch()
"
fi

echo "📄 Creating fairseq TSV manifests..."
python3 create_splits.py


# Step 3: Prepare audio features (CORRECTED)
echo "🎵 Step 3: Preparing audio features with k-means clustering..."
mkdir -p "$FEATURES_DIR"
mkdir -p "$FEATURES_DIR/mfcc"

for split in train valid;
do
    python3 "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py" \
        "$DATA_DIR/manifests" --split $split \
        --save-dir "$FEATURES_DIR" --checkpoint "$MODEL_PATH" --layer $LAYER

    # Generate MFCC features for both splits
    python3 "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py" \
        "$DATA_DIR/manifests" $split 1 0 "$FEATURES_DIR/mfcc"
done

# Learn the k-means model ONLY on the training data's MFCCs
echo "Training k-means model on training data..."
python3 "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py" \
    "$FEATURES_DIR/mfcc/train_0_1.npy" "$FEATURES_DIR/cls$CLUSTERS.km" $CLUSTERS \
    --percent 0.1 # Use a subset for faster training

# Generate pseudo-labels (.km files) for BOTH train and valid splits using the trained model
echo "Generating pseudo-labels for train and valid sets..."
for split in train valid;
do
    python3 "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py" \
        "$FEATURES_DIR/mfcc/${split}_0_1.npy" "$FEATURES_DIR/cls$CLUSTERS.km" "$FEATURES_DIR/${split}.km"
done

# --- Symbolic linking (CORRECTED) ---
echo "Creating symbolic links..."
ln -sf "$WORKSPACE_DIR/$FEATURES_DIR/train.npy" "$WORKSPACE_DIR/$DATA_DIR/manifests/train.npy"
ln -sf "$WORKSPACE_DIR/$FEATURES_DIR/valid.npy" "$WORKSPACE_DIR/$DATA_DIR/manifests/valid.npy"

ln -sf "$WORKSPACE_DIR/$FEATURES_DIR/train.km" "$WORKSPACE_DIR/$DATA_DIR/manifests/train.km"
ln -sf "$WORKSPACE_DIR/$FEATURES_DIR/valid.km" "$WORKSPACE_DIR/$DATA_DIR/manifests/valid.km"

ln -sf "$WORKSPACE_DIR/$FEATURES_DIR/train.lengths" "$WORKSPACE_DIR/$DATA_DIR/manifests/train.lengths"
ln -sf "$WORKSPACE_DIR/$FEATURES_DIR/valid.lengths" "$WORKSPACE_DIR/$DATA_DIR/manifests/valid.lengths"


# Step 4: Prepare text data
echo "📝 Step 4: Preparing Fongbe text data..."
TEXT_CORPUS="$DATA_DIR/combined_text.txt"
if [ ! -f "$TEXT_CORPUS" ]; then
    find "$DATA_DIR/text" -name "*.txt" -exec cat {} \; > "$TEXT_CORPUS"
fi
mkdir -p "$TEXT_FEATURES_DIR/phones"

# Create simple phoneme files for text processing
echo "Creating phoneme dictionary and text files..."
python3 -c "
import os, re

# Create basic phone dictionary
phones = ['a', 'e', 'i', 'o', 'u', 'ɛ', 'ɔ', 'b', 'p', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 'h', 'j', 'w', 'l', 'r', 'n', 'm', 'ŋ', 'SIL']
with open('$TEXT_FEATURES_DIR/phones/dict.phn.txt', 'w') as f:
    for i, phone in enumerate(phones):
        f.write(f'{phone} {i}\\n')

# Create simple phonemized text from the corpus
with open('$DATA_DIR/combined_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Basic text normalization and phonemization 
normalized_lines = []
for line in text.split('\\n'):
    line = line.strip().lower()
    if line:
        # Simple character-to-phone mapping
        phones_line = ' '.join(list(line.replace(' ', ' SIL ')))
        normalized_lines.append(phones_line)

# Save phonemized text
with open('$TEXT_FEATURES_DIR/lm.phones.sil.txt', 'w') as f:
    for line in normalized_lines:
        f.write(line + '\\n')

print(f'Created phoneme dictionary with {len(phones)} phones')
print(f'Processed {len(normalized_lines)} text lines')
"
python3 "$FAIRSEQ_ROOT/fairseq_cli/preprocess.py" \
    --dataset-impl mmap \
    --trainpref "$TEXT_FEATURES_DIR/lm.phones.sil.txt" \
    --workers 10 \
    --only-source \
    --destdir "$TEXT_FEATURES_DIR/phones" \
    --srcdict "$TEXT_FEATURES_DIR/phones/dict.phn.txt"

cp "$TEXT_FEATURES_DIR/phones/dict.phn.txt" "$DATA_DIR/manifests/dict.phn.txt"

# Step 5: Run Training
echo "🚀 Step 5: Starting wav2vec-U 2.0 GAN training..."
CHECKPOINT_DIR="$WORKSPACE_DIR/fongbe_checkpoints_official"
mkdir -p "$CHECKPOINT_DIR"

# Install wandb if not already installed
pip install wandb

# Initialize WandB (you'll need to login first time)
echo "🔑 Setting up WandB logging..."
echo "If this is your first time, you'll need to login to WandB"

fairseq-hydra-train \
    --config-dir "$WORKSPACE_DIR" \
    --config-name "fongbe_training_config" \
    task.data="$WORKSPACE_DIR/$DATA_DIR/manifests" \
    task.text_data="$WORKSPACE_DIR/$TEXT_FEATURES_DIR/phones" \
    task.kenlm_path=null \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    checkpoint.save_dir="$CHECKPOINT_DIR" \
    common.wandb_project="fongbe-wav2vec-u2" \
    model.target_dim=$(wc -l < "$TEXT_FEATURES_DIR/phones/dict.phn.txt") \
        optimization.max_update=100000 \
    dataset.batch_size=160 \
    dataset.validate_interval=1000

echo "✅ Training pipeline complete!"
