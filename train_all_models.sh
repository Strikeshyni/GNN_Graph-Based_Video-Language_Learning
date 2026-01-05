#!/bin/bash
# Script pour entraîner les 4 types de GNN en moins d'une heure chacun
# Utilise des features aléatoires (pas de vidéos réelles)

echo "=============================================="
echo "🚀 Entraînement Rapide des 4 Types de GNN"
echo "=============================================="
echo ""
echo "Temps estimé: ~15-20 minutes par modèle avec GPU"
echo "             ~4h total avec les 4 modèles"
echo ""

# Configuration optimale pour un entrainement rapide mais pertinent
# - 10 époques: suffisant pour voir les tendances
# - batch_size 64: bon équilibre vitesse/mémoire
# - lr 5e-4: convergence plus rapide

EPOCHS=50
BATCH_SIZE=32
LR=0.0001

echo "Configuration:"
echo "  - Époques: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LR"
echo ""

# 1. GAT (Graph Attention Network)
echo "=============================================="
echo "1/4 - Entraînement GAT..."
echo "=============================================="
python3 train.py \
    --gnn_type GAT \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_workers 8 \
    --gnn_layers 2 \
    --gnn_heads 4 \
    --exp_name avqa_gnn_gat \
    --resume ./checkpoints/avqa_gnn_gat_best.pth
echo ""

# 2. GCN (Graph Convolutional Network)
echo "=============================================="
echo "2/4 - Entraînement GCN..."
echo "=============================================="
python3 train.py \
    --gnn_type GCN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_workers 8 \
    --gnn_layers 2 \
    --exp_name avqa_gnn_gcn \
    --resume ./checkpoints/avqa_gnn_gcn_best.pth

echo ""

# 3. GraphSAGE
echo "=============================================="
echo "3/4 - Entraînement GraphSAGE..."
echo "=============================================="
python3 train.py \
    --gnn_type GraphSAGE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_workers 8 \
    --gnn_layers 2 \
    --exp_name avqa_gnn_graphsage \
    --resume ./checkpoints/avqa_gnn_graphsage_best.pth

echo ""

# 4. GIN (Graph Isomorphism Network)
echo "=============================================="
echo "4/4 - Entraînement GIN..."
echo "=============================================="
python3 train.py \
    --gnn_type GIN \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --num_workers 8 \
    --gnn_layers 2 \
    --exp_name avqa_gnn_gin \
    --resume ./checkpoints/avqa_gnn_gin_best.pth

echo ""
echo "Résultats sauvegardés dans:"
echo "  - ./checkpoints/avqa_gnn_gat_best.pth"
echo "  - ./checkpoints/avqa_gnn_gcn_best.pth"
echo "  - ./checkpoints/avqa_gnn_graphsage_best.pth"
echo "  - ./checkpoints/avqa_gnn_gin_best.pth"
echo ""
echo "Pour comparer les résultats:"
echo "  python3 evaluate.py --compare_architectures"
