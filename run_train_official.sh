#!/bin/bash
# Official Fongbe wav2vec-U 2.0 ASR Training Pipeline using fairseq scripts
# This script follows the official fairseq wav2vec unsupervised workflow

set -e

# Configuration
DATA_DIR="fongbe_data"
G2P_DICT="codes/dictionary.txt"
MODEL_PATH="xlsr_53_56k.pt"
FEATURES_DIR="fongbe_features_official"
TEXT_FEATURES_DIR="fongbe_text_features"
LANGUAGE="fongbe"  # Language code for Fongbe
CLUSTERS=64        # Number of clusters for wav2vec-U 2.0
LAYER=14           # Layer to extract from (0-based indexing)
MIN_PHONES=10      # Minimum phone observations to keep
SIL_PROB=0.5       # Silence probability for wav2vec-U 2.0

echo "🎯 Official Fongbe wav2vec-U 2.0 ASR Training Pipeline"
echo "====================================================="

# Check if environment is activated
ENV_NAME=$(basename "$CONDA_DEFAULT_ENV" 2>/dev/null || echo "")
if [[ "$ENV_NAME" != "fongbe_asr" ]]; then
    echo "❌ Please activate the fongbe_asr environment first:"
    echo "   source /opt/miniforge3/bin/activate fongbe_asr"
    exit 1
fi

# Set required environment variables
export FAIRSEQ_ROOT="/workspace/fairseq"

# Check for required environment variables and tools (not needed for custom Fongbe text processing)
# KENLM_ROOT and KALDI_ROOT are only needed for the official prepare_text.sh
# Our custom prepare_fongbe_text.sh creates fairseq binary files directly

# Phonemizer and espeak not needed since we use G2P dictionary
# Our custom prepare_fongbe_text.sh uses the G2P dictionary directly

# Check and download test data if needed
if [ ! -d "$DATA_DIR/audio" ] || [ ! -d "$DATA_DIR/text" ] || [ -z "$(ls -A $DATA_DIR/audio 2>/dev/null)" ]; then
    echo "📥 Test data not found. Downloading from HuggingFace..."
    python download_test_data.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download test data"
        exit 1
    fi
fi

if [ ! -f "$G2P_DICT" ]; then
    echo "❌ G2P dictionary not found: $G2P_DICT"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ wav2vec 2.0 model not found: $MODEL_PATH"
    echo "📥 Downloading XLSR-53 model..."
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt
fi

# Step 1: Prepare audio data and create manifests
echo ""
echo "📂 Step 1: Converting audio to 16kHz mono and creating manifests..."
mkdir -p "$DATA_DIR/manifests"
mkdir -p "$DATA_DIR/audio_16k"

# Convert audio to 16kHz mono if not already done
if [ ! -f "$DATA_DIR/audio_16k/.converted" ]; then
    echo "🔄 Converting audio files to 16kHz mono..."
    python3 -c "
import os
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

audio_dir = '$DATA_DIR/audio'
output_dir = '$DATA_DIR/audio_16k'
os.makedirs(output_dir, exist_ok=True)

for f in os.listdir(audio_dir):
    if f.endswith('.wav'):
        input_path = os.path.join(audio_dir, f)
        output_path = os.path.join(output_dir, f)
        
        if not os.path.exists(output_path):
            print(f'Converting {f}...')
            # Load with librosa to handle resampling and mono conversion
            audio, sr = librosa.load(input_path, sr=16000, mono=True)
            # Save as 16kHz mono
            sf.write(output_path, audio, 16000)
        else:
            print(f'Skipping {f} (already exists)')

# Create marker file
Path(os.path.join(output_dir, '.converted')).touch()
print('Audio conversion complete!')
"
else
    echo "✅ Audio already converted to 16kHz mono"
fi

# Create proper TSV manifests with tab-separated format
echo "📄 Creating fairseq TSV manifests..."
python3 -c "
import os
import soundfile as sf

audio_dir = '$DATA_DIR/audio_16k'
manifest_dir = '$DATA_DIR/manifests'
os.makedirs(manifest_dir, exist_ok=True)

# Get all wav files
wav_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
wav_files = sorted(wav_files)  # Sort for consistency

# Create train.tsv (use first few files for testing)
train_files = wav_files[:3]  # Limit to 3 files for testing
with open(os.path.join(manifest_dir, 'train.tsv'), 'w') as f:
    # Write root directory path
    f.write(os.path.abspath(audio_dir) + '\n')
    
    # Write file paths with tab separation (format: filename\tframes)
    for fname in train_files:
        full_path = os.path.join(audio_dir, fname)
        try:
            info = sf.info(full_path)
            frames = info.frames
            f.write(f'{fname}\t{frames}\n')
        except Exception as e:
            print(f'Error processing {fname}: {e}')

# Copy for validation (same data for now)
with open(os.path.join(manifest_dir, 'valid.tsv'), 'w') as f_out:
    with open(os.path.join(manifest_dir, 'train.tsv'), 'r') as f_in:
        f_out.write(f_in.read())

print(f'Created manifests with {len(train_files)} files')
"

echo "✅ Created data manifests and converted audio to 16kHz mono"

# Step 2: Prepare audio features using official fairseq script
echo ""
echo "🎵 Step 2: Preparing audio features with k-means clustering..."
echo "   Using $CLUSTERS clusters from layer $LAYER"

mkdir -p "$FEATURES_DIR"

# Use official prepare_audio_v2.sh script for wav2vec-U 2.0
# Use official prepare_audio_v2.sh but with manual k-means clustering
echo "🔧 Step 2a: Extracting wav2vec 2.0 features..."
for split in train valid; do
    python "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py" \
        "$DATA_DIR/manifests" --split $split \
        --save-dir "$FEATURES_DIR" --checkpoint "$MODEL_PATH" --layer $LAYER
done

echo "🔧 Step 2b: Extracting MFCC features for k-means clustering..."
python "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py" \
    "$FEATURES_DIR" train 1 0 "$FEATURES_DIR/mfcc"

echo "🔧 Step 2c: Training k-means model with $CLUSTERS clusters..."
python "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py" \
    "$FEATURES_DIR/mfcc" train 1 "$FEATURES_DIR/mfcc/cls$CLUSTERS" $CLUSTERS \
    --init k-means++ --max_iter 100 --batch_size 10000 --tol 0.0 --max_no_improvement 100 --n_init 20 --reassignment_ratio 0.5

echo "🔧 Step 2d: Applying k-means labels..."
python "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py" \
    "$FEATURES_DIR/mfcc" train "$FEATURES_DIR/mfcc/cls$CLUSTERS" 1 0 "$FEATURES_DIR/mfcc/cls${CLUSTERS}_idx"

# Copy the k-means labels to the expected location
cp "$FEATURES_DIR/mfcc/cls${CLUSTERS}_idx/train_0_1.km" "$FEATURES_DIR/train.km"

if [ $? -ne 0 ]; then
    echo "❌ Failed to prepare audio features"
    exit 1
fi

echo "✅ Audio features prepared with k-means clustering"

# Step 3: Prepare text data using custom Fongbe script
echo ""
echo "📝 Step 3: Preparing Fongbe text data..."

# Combine all text files into a single corpus
TEXT_CORPUS="$DATA_DIR/combined_text.txt"
if [ ! -f "$TEXT_CORPUS" ]; then
    echo "📚 Combining text files into corpus..."
    find "$DATA_DIR/text" -name "*.txt" -exec cat {} \; > "$TEXT_CORPUS"
    echo "✅ Combined $(wc -l < "$TEXT_CORPUS") lines of text"
fi

# Use custom Fongbe text preparation script
mkdir -p "$TEXT_FEATURES_DIR"

bash prepare_fongbe_text.sh \
    "$TEXT_CORPUS" \
    "$TEXT_FEATURES_DIR" \
    "$G2P_DICT" \
    $MIN_PHONES \
    $SIL_PROB

if [ $? -ne 0 ]; then
    echo "❌ Failed to prepare Fongbe text data"
    exit 1
fi

echo "✅ Fongbe text data prepared with G2P dictionary and phonemization"

# Step 4: Run GAN training using official fairseq command
echo ""
echo "🚀 Step 4: Starting wav2vec-U 2.0 GAN training..."

# Set up training parameters
PREFIX="fongbe_w2vu2_gan"
CONFIG_NAME="w2vu2"
TASK_DATA="$FEATURES_DIR"
TEXT_DATA="$TEXT_FEATURES_DIR/phones"
KENLM_PATH="$TEXT_FEATURES_DIR/phones/train.bin"  # Use fairseq preprocessed binary

# Create checkpoint directory
CHECKPOINT_DIR="fongbe_checkpoints_official"
mkdir -p "$CHECKPOINT_DIR"

echo "🔧 Training configuration:"
echo "   • Audio features: $TASK_DATA"
echo "   • Text data: $TEXT_DATA"  
echo "   • Language model: $KENLM_PATH"
echo "   • Checkpoints: $CHECKPOINT_DIR"
echo ""

# Run GAN training with official fairseq-hydra-train (override launcher config)
PYTHONPATH="$FAIRSEQ_ROOT" PREFIX="$PREFIX" fairseq-hydra-train \
    --config-dir "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/gan" \
    --config-name "$CONFIG_NAME" \
    task.data="$TASK_DATA" \
    task.text_data="$TEXT_DATA" \
    task.kenlm_path="$KENLM_PATH" \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    checkpoint.save_dir="$CHECKPOINT_DIR" \
    common.tensorboard_logdir="fongbe_tensorboard_official" \
    model.target_dim=$CLUSTERS \
    optimization.max_update=10000 \
    dataset.batch_size=16 \
    dataset.validate_interval=1000 \
    hydra.launcher._target_=hydra._internal.BasicLauncher

if [ $? -ne 0 ]; then
    echo "❌ GAN training failed"
    exit 1
fi

echo ""
echo "✅ Official wav2vec-U 2.0 training pipeline complete!"
echo ""
echo "📊 Training artifacts:"
echo "   • Audio features: $FEATURES_DIR"
echo "   • Text features: $TEXT_FEATURES_DIR"  
echo "   • Checkpoints: $CHECKPOINT_DIR"
echo "   • Tensorboard logs: fongbe_tensorboard_official"
echo ""
echo "📈 Monitor training with:"
echo "   tensorboard --logdir fongbe_tensorboard_official"
echo ""
echo "🔍 Generate predictions with:"
echo "   python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/w2vu_generate.py \\"
echo "     --config-dir $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/config/generate \\"
echo "     --config-name viterbi \\"
echo "     fairseq.common.user_dir=$FAIRSEQ_ROOT/examples/wav2vec/unsupervised \\"
echo "     fairseq.task.data=$TASK_DATA \\"
echo "     fairseq.common_eval.path=$CHECKPOINT_DIR/checkpoint_best.pt \\"
echo "     fairseq.dataset.gen_subset=valid \\"
echo "     results_path=fongbe_predictions"