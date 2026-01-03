"""
AVQA-GNN: Graph Neural Networks for Audio-Visual Question Answering
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.models.avqa_gnn import AVQA_GNN
from src.models.gat import GAT, GNNStack
from src.models.alternative_gnns import create_gnn_stack

__all__ = [
    'AVQA_GNN',
    'GAT',
    'GNNStack',
    'create_gnn_stack'
]
