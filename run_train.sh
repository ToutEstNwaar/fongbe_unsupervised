#!/bin/bash
# Complete Fongbe wav2vec-U 2.0 ASR Training Pipeline
# This script runs the complete pipeline from raw data to trained model

set -e

# Configuration
DATA_DIR="fongbe_data"
G2P_DICT="codes/dictionary.txt"
MODEL_PATH="xlsr_53_56k.pt"
FEATURES_DIR="fongbe_features"

echo "🎯 Fongbe wav2vec-U 2.0 ASR Training Pipeline"
echo "============================================"

# Check if environment is activated
ENV_NAME=$(basename "$CONDA_DEFAULT_ENV" 2>/dev/null || echo "")
if [[ "$ENV_NAME" != "fongbe_asr" ]]; then
    echo "❌ Please activate the fongbe_asr environment first:"
    echo "   source /opt/miniforge3/bin/activate fongbe_asr"
    exit 1
fi

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
    echo "Downloading XLSR-53 model..."
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt
fi

echo "📝 Step 1: Processing Fongbe text data..."
python process_fongbe_data.py "$DATA_DIR/text" "$G2P_DICT"

echo ""
echo "🎵 Step 2: Extracting wav2vec 2.0 features..."
python extract_features.py "$DATA_DIR/audio" "$MODEL_PATH" "$FEATURES_DIR"

echo ""
echo "🚀 Step 3: Starting wav2vec-U 2.0 training..."
python train_fongbe_wav2vec_u2.py "$FEATURES_DIR"

echo ""
echo "✅ Training pipeline complete!"
echo ""
echo "📊 Training artifacts:"
echo "   • Logs: /workspace/fongbe_tensorboard"
echo "   • Checkpoints: /workspace/fongbe_checkpoints"
echo "   • Features: /workspace/$FEATURES_DIR"
echo ""
echo "📈 Monitor training with:"
echo "   tensorboard --logdir /workspace/fongbe_tensorboard"
