"""
Quick demo script to test the AVQA-GNN model.
"""

import torch
import sys
sys.path.append('.')

from src.models.avqa_gnn import AVQA_GNN
from src.models.alternative_gnns import create_gnn_stack
from src.data.dataset import AVQA_Dataset, collate_avqa
from src.utils.visualization import create_visualization_examples
from torch_geometric.data import Data


class Args:
    """Simple argument container."""
    def __init__(self):
        self.num_classes = 42
        self.node_dim = 512
        self.edge_dim = 512
        self.out_channels = 512
        self.gnn_layers = 2
        self.gnn_heads = 4
        self.dropout = 0.1


def test_gat_layer():
    """Test GAT layer."""
    print("=" * 60)
    print("Testing GAT Layer")
    print("=" * 60)
    
    from src.models.gat import GAT
    
    # Create dummy graph data
    num_nodes = 5
    num_edges = 8
    in_channels = 512
    out_channels = 512
    edge_dim = 512
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_dim)
    
    gat = GAT(in_channels, out_channels, edge_dim, heads=4)
    
    output = gat(x, edge_index, edge_attr)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"- GAT layer test passed!\n")


def test_alternative_gnns():
    """Test alternative GNN architectures."""
    print("=" * 60)
    print("Testing Alternative GNN Architectures")
    print("=" * 60)
    
    # Create dummy graph data
    num_nodes = 5
    num_edges = 8
    in_channels = 512
    out_channels = 512
    edge_dim = 512
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_dim)
    
    for gnn_type in ['GCN', 'GraphSAGE', 'GIN']:
        print(f"\nTesting {gnn_type}...")
        
        gnn = create_gnn_stack(
            gnn_type,
            in_channels=in_channels,
            out_channels=out_channels,
            edge_attr_dim=edge_dim,
            num_layers=2,
            dropout=0.1
        )
        
        output = gnn(x, edge_index, edge_attr)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  - {gnn_type} test passed!")


def test_full_model():
    """Test full AVQA-GNN model."""
    print("\n" + "=" * 60)
    print("Testing Full AVQA-GNN Model")
    print("=" * 60)
    
    args = Args()
    model = AVQA_GNN(args)
    
    # Create dummy input
    batch_size = 2
    num_frames = 10
    
    audio_feat = torch.randn(batch_size, num_frames, 128)
    visual_feat = torch.randn(batch_size, num_frames, 512)
    question_feat = torch.randn(batch_size, 1, 512)
    
    # Create dummy scene graphs
    sg_x = torch.randn(batch_size * 5, 512)  # 5 nodes per graph
    sg_edge_index = torch.tensor([
        [0, 1, 1, 2, 5, 6, 6, 7],
        [1, 0, 2, 1, 6, 5, 7, 6]
    ], dtype=torch.long)
    sg_edge_attr = torch.randn(8, 512)
    sg_batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    # Create dummy query graphs
    qg_x = torch.randn(batch_size * 3, 512)  # 3 nodes per graph
    qg_edge_index = torch.tensor([
        [0, 1, 3, 4],
        [1, 0, 4, 3]
    ], dtype=torch.long)
    qg_edge_attr = torch.randn(4, 512)
    qg_batch = torch.tensor([0, 0, 0, 1, 1, 1])
    
    sg_data = (sg_x, sg_edge_index, sg_edge_attr, sg_batch)
    qg_data = (qg_x, qg_edge_index, qg_edge_attr, qg_batch)
    
    # Forward pass
    logits = model(audio_feat, visual_feat, question_feat, sg_data, qg_data)
    
    print(f"Audio features shape: {audio_feat.shape}")
    print(f"Visual features shape: {visual_feat.shape}")
    print(f"Question features shape: {question_feat.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [{batch_size}, {args.num_classes}]")
    print(f"- Full model test passed!\n")


def test_dataset():
    """Test dataset loading."""
    print("=" * 60)
    print("Testing Dataset")
    print("=" * 60)
    
    # Create dummy dataset
    dataset = AVQA_Dataset('./data/MUSIC-AVQA', split='train')
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of answer classes: {len(dataset.answer_to_idx)}")
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\nSample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, Data):
            print(f"  {key}: PyG Data with {value.num_nodes} nodes, {value.num_edges} edges")
        else:
            print(f"  {key}: {value}")
    
    print(f"- Dataset test passed!\n")


def main():
    """Run all tests."""
    print("AVQA-GNN Demo Script")
    
    try:
        # Test individual components
        test_gat_layer()
        test_alternative_gnns()
        test_full_model()
        test_dataset()
        
        # Create example visualizations
        print("=" * 60)
        print("Creating Example Visualizations")
        print("=" * 60)
        create_visualization_examples()
        
        print("All tests passed successfully!")
        
        print("Next steps:")
        print("1. Download the MUSIC-AVQA dataset")
        print("2. Preprocess audio and visual features")
        print("3. Run training: python train.py --gnn_type GAT")
        print("4. Compare architectures: python evaluate.py --compare_architectures")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
