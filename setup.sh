#!/bin/bash

# Setup script for AVQA-GNN project
# This script installs all dependencies and sets up the environment

echo "=========================================="
echo "AVQA-GNN Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python >= 3.8
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python version is sufficient"
else
    echo "✗ Python 3.8+ is required. Please upgrade Python."
    exit 1
fi

echo ""
echo "Installing dependencies..."
echo "=========================================="

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo ""
echo "Installing PyTorch..."
echo "Note: Adjust CUDA version in requirements.txt if needed"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo ""
echo "Installing PyTorch Geometric..."
pip install torch-geometric

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python3 -m spacy download en_core_web_sm

# Install SceneGraphParser
echo ""
echo "Installing SceneGraphParser..."
pip install git+https://github.com/vacancy/SceneGraphParser.git

# Create necessary directories
echo ""
echo "Creating project directories..."
mkdir -p data/MUSIC-AVQA
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p visualizations

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download MUSIC-AVQA dataset to data/MUSIC-AVQA/"
echo "2. Run demo: python demo.py"
echo "3. Train model: python train.py --gnn_type GAT"
echo "4. Evaluate: python evaluate.py --compare_architectures"
echo ""
echo "For more information, see README.md"
echo ""
