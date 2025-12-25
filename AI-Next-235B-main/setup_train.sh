#!/bin/bash
# Script to install dependencies and prepare for training on The Pile dataset

echo "Installing required packages..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets tokenizers
pip install tqdm numpy matplotlib
pip install deepspeed
pip install wandb  # For experiment tracking
pip install fairscale  # For additional distributed training features

echo "Dependencies installed successfully!"

echo "Setting up data directory for The Pile dataset..."
mkdir -p /workspace/pile_data

echo "To download The Pile dataset, you would typically need to:"
echo "1. Go to https://the-eye.eu/public/AI/pile/"
echo "2. Download the desired .jsonl files"
echo "3. Place them in /workspace/pile_data/"

echo "For testing purposes, we'll create a small sample dataset:"
mkdir -p /workspace/pile_data
cat << EOF > /workspace/pile_data/test_sample.jsonl
{"text": "The quick brown fox jumps over the lazy dog. This is a sample sentence for training."}
{"text": "Neural networks are powerful computational models inspired by the human brain."}
{"text": "Deep learning has revolutionized artificial intelligence in recent years."}
{"text": "Transformer architectures have become the foundation for modern language models."}
{"text": "The Pile is a large text dataset used for training advanced language models."}
EOF

echo "Sample dataset created for testing."

echo "To start training, run:"
echo "python train_pile.py"