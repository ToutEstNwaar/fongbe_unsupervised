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
CLUSTERS=34
LAYER=14
MIN_PHONES=10
SIL_PROB=0.5

WORKSPACE_DIR="/workspace/fongbe_unsupervised"

echo "🎯 Unified Fongbe wav2vec-U 2.0 ASR Training Pipeline"
echo "====================================================="

# Activate conda environment
source /opt/miniforge3/bin/activate fongbe_asr

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
python3 -c "
import os, soundfile as sf
audio_dir = '$DATA_DIR/audio_16k'
manifest_dir = '$DATA_DIR/manifests'
os.makedirs(manifest_dir, exist_ok=True)
wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
train_files = wav_files[:3]
with open(os.path.join(manifest_dir, 'train.tsv'), 'w') as f:
    f.write(os.path.abspath(audio_dir) + '\n')
    for fname in train_files:
        info = sf.info(os.path.join(audio_dir, fname))
        f.write(f'{fname}\t{info.frames}\n')
with open(os.path.join(manifest_dir, 'valid.tsv'), 'w') as f_out, open(os.path.join(manifest_dir, 'train.tsv'), 'r') as f_in:
    f_out.write(f_in.read())
"

# Step 3: Prepare audio features
echo "🎵 Step 3: Preparing audio features with k-means clustering..."
mkdir -p "$FEATURES_DIR"
for split in train valid;
do
    python3 "$FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py" \
        "$DATA_DIR/manifests" --split $split \
        --save-dir "$FEATURES_DIR" --checkpoint "$MODEL_PATH" --layer $LAYER
done
python3 "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py" \
    "$FEATURES_DIR" train 1 0 "$FEATURES_DIR/mfcc"
python3 "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py" \
    "$FEATURES_DIR/mfcc" train 1 "$FEATURES_DIR/mfcc/cls$CLUSTERS" $CLUSTERS \
    --init k-means++ --max_iter 100 --batch_size 10000 --tol 0.0 --max_no_improvement 100 --n_init 20 --reassignment_ratio 0.5
python3 "$FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py" \
    "$FEATURES_DIR/mfcc" train "$FEATURES_DIR/mfcc/cls$CLUSTERS" 1 0 "$FEATURES_DIR/mfcc/cls${CLUSTERS}_idx"
cp "$FEATURES_DIR/mfcc/cls${CLUSTERS}_idx/train_0_1.km" "$FEATURES_DIR/train.km"

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
python3 -c "
import sys, re
from collections import Counter
def normalize_fongbe_text(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s'']', '', text)
    text = re.sub(r'\d+', '', text)
    return text
with open('$TEXT_CORPUS', 'r', encoding='utf-8') as f, open('$TEXT_FEATURES_DIR/lm.upper.lid.txt', 'w') as f_out:
    for line in f:
        line = line.strip()
        if line:
            normalized = normalize_fongbe_text(line)
            if normalized and len(normalized.split()) >= 2:
                f_out.write(normalized + '\n')
word_counts = Counter()
with open('$TEXT_FEATURES_DIR/lm.upper.lid.txt', 'r') as f:
    for line in f:
        for word in line.strip().split():
            word_counts[word] += 1
with open('$TEXT_FEATURES_DIR/dict.txt', 'w') as f:
    for word, count in word_counts.most_common():
        if count >= 2:
            f.write(f'{word} {count}\n')
"
python3 -c "
import sys, re
def load_g2p_dictionary(dict_path):
    g2p_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                grapheme, phoneme = line.strip().split('\t', 1)
                g2p_dict[grapheme] = phoneme
    return g2p_dict
def apply_g2p_rules(word, g2p_dict):
    word = re.sub(r'([aeiouɛɔ])n([ptk])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])m([pb])', r'\1̃\2', word)
    word = re.sub(r'([aeiouɛɔ])n$', r'\1̃', word)
    word = re.sub(r'([aeiouɛɔ])m$', r'\1̃', word)
    result = []
    i = 0
    while i < len(word):
        found = False
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
g2p_dict = load_g2p_dictionary('$G2P_DICT')
with open('$TEXT_FEATURES_DIR/dict.txt', 'r') as f_in, open('$TEXT_FEATURES_DIR/phones.txt', 'w') as f_out:
    for line in f_in:
        word = line.strip().split()[0]
        phones = apply_g2p_rules(word, g2p_dict)
        f_out.write(phones + '\n')
"
paste "$TEXT_FEATURES_DIR/dict.txt" "$TEXT_FEATURES_DIR/phones.txt" > "$TEXT_FEATURES_DIR/lexicon.lst"
python3 -c "
from collections import Counter
phone_counts = Counter()
with open('$TEXT_FEATURES_DIR/phones.txt', 'r') as f:
    for line in f:
        for phone in line.strip().split():
            phone_counts[phone] += 1
with open('$TEXT_FEATURES_DIR/phones/dict.txt', 'w') as f:
    for phone, count in phone_counts.most_common():
        if count >= $MIN_PHONES:
            f.write(f'{phone} {count}\n')
"
cp "$TEXT_FEATURES_DIR/phones/dict.txt" "$TEXT_FEATURES_DIR/phones/dict.phn.txt"
echo "<SIL> 0" >> "$TEXT_FEATURES_DIR/phones/dict.phn.txt"
python3 "$FAIRSEQ_ROOT/fairseq_cli/preprocess.py" \
    --dataset-impl mmap \
    --trainpref "$TEXT_FEATURES_DIR/lm.upper.lid.txt" \
    --workers 10 \
    --only-source \
    --destdir "$TEXT_FEATURES_DIR/phones" \
    --srcdict "$TEXT_FEATURES_DIR/phones/dict.phn.txt"

cp "$TEXT_FEATURES_DIR/phones/dict.phn.txt" "$DATA_DIR/manifests/dict.phn.txt"

# Step 5: Run Training
echo "🚀 Step 5: Starting wav2vec-U 2.0 GAN training..."
CHECKPOINT_DIR="$WORKSPACE_DIR/fongbe_checkpoints_official"
mkdir -p "$CHECKPOINT_DIR"
fairseq-hydra-train \
    --config-dir "$WORKSPACE_DIR" \
    --config-name "fongbe_training_config" \
    task.data="$WORKSPACE_DIR/$DATA_DIR/manifests" \
    task.text_data="$WORKSPACE_DIR/$TEXT_FEATURES_DIR/phones" \
    task.kenlm_path=null \
    common.user_dir="$FAIRSEQ_ROOT/examples/wav2vec/unsupervised" \
    checkpoint.save_dir="$CHECKPOINT_DIR" \
    common.tensorboard_logdir="$WORKSPACE_DIR/fongbe_tensorboard_official" \
    model.target_dim=$(wc -l < "$TEXT_FEATURES_DIR/phones/dict.phn.txt") \
        optimization.max_update=20 \
    dataset.batch_size=1 \
    dataset.validate_interval=1000

echo "✅ Training pipeline complete!"
