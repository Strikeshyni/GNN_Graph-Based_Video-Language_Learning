# Graph-Based Video-Language Learning for Audio-Visual Question Answering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Implementation of Graph Neural Networks for Audio-Visual Question Answering (AVQA), based on the paper "Graph-Based Video-Language Learning with Multi-Grained Audio-Visual Alignment".

**Original Article:** [Stanford CS224W Blog Post](https://medium.com/stanford-cs224w/graph-based-video-language-learning-5efd51561640)

**Dataset:** [MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Modifications](#key-modifications)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Architecture Details](#architecture-details)
- [Citation](#citation)

---

## Overview

This project implements and extends the Graph-Based Video-Language Learning approach for Audio-Visual Question Answering. The model combines:

- **Graph Neural Networks (GNNs)** for encoding scene graphs and query graphs
- **Multi-Grained Alignment (MgA)** for audio-visual feature fusion
- **Hierarchical Matching** for aligning multimodal representations
- **Cross-Modal Attention** for capturing audio-visual interactions

### Task Description

Given a video with audio and a question about its content, the model must select the correct answer from multiple choices. Questions can be about:
- Visual content ("What instrument is visible?")
- Audio content ("What sound is playing?")
- Audio-visual interactions ("Which instrument is making this sound?")
- Spatial/temporal reasoning ("Where is the performer?", "When does the sound occur?")

---

## Project Structure

```
GNN_Graph-Based_Video-Language_Learning/
├── src/
│   ├── models/
│   │   ├── gat.py                 # Graph Attention Network implementation
│   │   ├── avqa_gnn.py           # Complete AVQA-GNN model
│   │   └── alternative_gnns.py    # GCN, GraphSAGE, GIN implementations
│   ├── data/
│   │   ├── scene_graph_parser.py  # Scene graph generation
│   │   └── dataset.py             # Dataset loader
│   └── utils/
│       └── visualization.py       # Visualization tools
├── given_code/                    # Original code snippets from article
├── experiments/                   # Experiment results
├── visualizations/               # Generated visualizations
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── demo.py                       # Demo and testing script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## Key Modifications

This implementation extends the original work with:

### 1. **Alternative GNN Architectures**
- **GCN (Graph Convolutional Network)**: Simpler baseline without attention
- **GraphSAGE**: Sample-and-aggregate approach for scalability
- **GIN (Graph Isomorphism Network)**: Theoretically more expressive
- Easy comparison through `--gnn_type` argument

### 2. **Comprehensive Visualization Tools**
- Cross-modal attention heatmaps
- Graph structure visualization with attention weights
- Training curves and performance metrics
- Architecture comparison charts
- Confusion matrices and per-question-type analysis

### 3. **Modular and Extensible Codebase**
- Clean separation of concerns
- Well-documented components
- Easy to extend with new architectures
- Flexible dataset interface

### 4. **Complete Training Pipeline**
- TensorBoard integration
- Checkpoint management
- Learning rate scheduling
- Gradient clipping
- Comprehensive logging

### 5. **Evaluation Framework**
- Ablation studies
- Architecture comparison
- Per-question-type analysis
- Confusion matrix generation
- Detailed classification reports

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd GNN_Graph-Based_Video-Language_Learning
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install PyTorch Geometric** (if not auto-installed)
```bash
pip install torch-geometric
```

4. **Download spaCy model** (for scene graph parsing)
```bash
python -m spacy download en_core_web_sm
```

5. **Install sng_parser**
```bash
pip install git+https://github.com/vacancy/SceneGraphParser.git
```

---

## Usage

### Quick Demo

Run the demo script to test all components:

```bash
python demo.py
```

This will:
- Test GAT and alternative GNN layers
- Test the full AVQA-GNN model
- Test dataset loading
- Generate example visualizations

### Training

Train the model with GAT (default):

```bash
python train.py \
    --data_dir ./data/MUSIC-AVQA \
    --gnn_type GAT \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --exp_name avqa_gnn_gat
```

Train with alternative architectures:

```bash
# GCN
python train.py --gnn_type GCN --exp_name avqa_gnn_gcn

# GraphSAGE
python train.py --gnn_type GraphSAGE --exp_name avqa_gnn_sage

# GIN
python train.py --gnn_type GIN --exp_name avqa_gnn_gin
```

### Evaluation

Evaluate a single model:

```bash
python evaluate.py \
    --checkpoint_path ./checkpoints/avqa_gnn_best.pth \
    --output_dir ./results
```

Compare all architectures:

```bash
python evaluate.py \
    --compare_architectures \
    --checkpoint_dir ./checkpoints \
    --output_dir ./results
```

Perform ablation study:

```bash
python evaluate.py \
    --ablation \
    --checkpoint_path ./checkpoints/avqa_gnn_best.pth \
    --output_dir ./results
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir ./logs
```

---

## Results

Empty for now

## Architecture Details

### Model Components

#### 1. **Scene Graph Encoder**
- Extracts structured representations from video frames
- Uses BLIP for image captioning (optional)
- Parses captions into scene graphs with entities and relations
- Encodes graphs with GAT/GCN/GraphSAGE/GIN

#### 2. **Query Graph Encoder**
- Parses questions into semantic graphs
- Identifies arguments and semantic roles
- Separate GNN encoder for query understanding

#### 3. **Joint Graph Representation**
- Computes similarity between scene and query graphs
- Generates aligned joint representation
- Uses bilinear fusion for combining features

#### 4. **Multi-Grained Alignment (MgA)**
- Parallel 1D convolutions with kernels {1, 3, 5}
- Captures multi-scale patterns in audio/visual sequences
- Cross-attention between modalities at each scale

#### 5. **Hierarchical Matching**
- Two-stage matching strategy
- First: combine features within same scale
- Second: fuse across scales
- Graph-guided attention weighting

#### 6. **Classification Head**
- Gated fusion of audio-visual features
- Question-guided feature modulation
- MLP classifier for answer prediction

### GAT with Edge Features

The core GAT implementation extends standard attention to include edge features:

```
α_ij = softmax(LeakyReLU(a^T [W_src h_i || W_dst h_j || W_edge e_ij]))
```

Where:
- `h_i, h_j`: source and target node features
- `e_ij`: edge features
- `W_*`: learnable weight matrices
- `a`: attention weight vector

---

## Citation

If you use this code, please cite:

```bibtex
@article{lyu2023graph,
  title={Graph-Based Video-Language Learning with Multi-Grained Audio-Visual Alignment},
  author={Lyu, Chenyang and Li, Wenxi and Ji, Tianbo and others},
  journal={ACM International Conference on Multimedia},
  year={2023}
}
```

Original article:
```bibtex
@article{wei2024graph,
  title={Graph-Based Video-Language Learning},
  author={Wei, Zhengyang and Dai, Tianyuan and Duan, Haoyi},
  journal={Stanford CS224W Blog},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Original paper authors: Chenyang Lyu, Wenxi Li, Tianbo Ji, et al.
- Stanford CS224W course and blog authors
- MUSIC-AVQA dataset creators
- PyTorch Geometric team

---

**Original code snippets from the article are preserved in `given_code/` directory.**
