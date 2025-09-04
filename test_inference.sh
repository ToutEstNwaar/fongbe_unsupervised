#!/bin/bash

# Activate conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate fongbe_asr

# Set paths
FAIRSEQ_ROOT="/workspace/fongbe_unsupervised/fairseq"
USER_DIR="/workspace/fongbe_unsupervised/fairseq/examples/wav2vec/unsupervised"
CHECKPOINT="/workspace/fongbe_unsupervised/fongbe_checkpoints_official/checkpoint_last.pt"
DATA_PATH="/workspace/fongbe_unsupervised/fongbe_data/manifests"
TEXT_DATA="/workspace/fongbe_unsupervised/fongbe_text_features/phones"

# Set PYTHONPATH to include fairseq root for proper imports
export PYTHONPATH="$FAIRSEQ_ROOT:$PYTHONPATH"

cd /workspace/fongbe_unsupervised

echo "🎯 Testing wav2vec-U 2.0 inference"
echo "============================================"

# Test using fairseq-generate
python $FAIRSEQ_ROOT/fairseq_cli/generate.py \
    $DATA_PATH \
    --user-dir $USER_DIR \
    --task unpaired_audio_text \
    --text-data $TEXT_DATA \
    --path $CHECKPOINT \
    --gen-subset valid \
    --max-tokens 1000000 \
    --results-path /workspace/fongbe_unsupervised/inference_results \
    --beam 1 \
    --batch-size 1

echo "✅ Inference test completed!"