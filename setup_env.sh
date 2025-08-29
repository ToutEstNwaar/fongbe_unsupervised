#!/bin/bash
# Fongbe wav2vec-U 2.0 ASR Training - Environment Setup
# This script sets up the complete environment for training

set -e

echo "🚀 Setting up Fongbe wav2vec-U 2.0 ASR training environment..."
echo "This will take 5-10 minutes to complete..."

# Create conda environment
echo ""
echo "📦 Creating conda environment 'fongbe_asr'..."
source /opt/miniforge3/bin/activate
conda create -n fongbe_asr python=3.8 -y
source /opt/miniforge3/bin/activate fongbe_asr

# Upgrade pip and install critical dependencies first (in order to avoid conflicts)
echo ""
echo "📦 Installing critical dependencies (avoiding conflicts)..."
pip install --upgrade pip==24.0
pip install omegaconf==2.0.6
pip install hydra-core==1.0.7
pip install "numpy<1.24"  # Must be installed before other packages

# Clone and install fairseq
if [ ! -d "fairseq" ]; then
    echo ""
    echo "📦 Cloning fairseq repository..."
    git clone https://github.com/facebookresearch/fairseq.git
fi

echo ""
echo "📦 Installing fairseq..."
cd fairseq
pip install -e .
cd ..

# Install PyTorch with CUDA support (critical to do this after fairseq)
echo ""
echo "📦 Installing PyTorch with CUDA support..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional required packages
echo ""
echo "📦 Installing additional packages..."
pip install tensorboardX>=2.6.0
pip install npy-append-array>=0.9.0
pip install soundfile>=0.12.1
pip install librosa>=0.11.0
pip install transformers>=4.46.0
pip install regex>=2024.0.0
pip install requests>=2.32.0

echo ""
echo "🧪 Testing installation..."

# Test critical imports
python -c "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import fairseq; print('✅ Fairseq installed')"
python -c "import librosa; print('✅ Librosa installed')"
python -c "import transformers; print('✅ Transformers installed')"
python -c "import soundfile; print('✅ Soundfile installed')"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📋 Summary:"
echo "   • Conda environment: fongbe_asr"
echo "   • Python: $(python --version)"
echo "   • PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "   • CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "🚀 To activate: source /opt/miniforge3/bin/activate fongbe_asr"
