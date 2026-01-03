"""
Alternative GNN architectures for comparison with GAT.
Includes GCN, GraphSAGE, and GIN implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d


class GCNStack(nn.Module):
    """
    Graph Convolutional Network (GCN) stack for graph encoding.
    Simpler than GAT but doesn't use attention mechanism.
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim, num_layers, 
                 dropout=0.0):
        super(GCNStack, self).__init__()
        
        assert num_layers >= 1, 'Number of layers must be >= 1'
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Edge feature projection
        self.edge_encoder = nn.Linear(edge_attr_dim, in_channels)
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, out_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(out_channels, out_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList([
            BatchNorm1d(out_channels) 
            for _ in range(num_layers - 1)
        ])
        
        # Post-processing MLP
        self.post_mp = Sequential(
            Linear(out_channels, out_channels),
            nn.Dropout(dropout),
            Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through GCN stack."""
        # Encode edge features by adding to node features
        row, col = edge_index
        edge_embeddings = self.edge_encoder(edge_attr)
        
        # Aggregate edge features to nodes
        edge_contribution = torch.zeros_like(x)
        edge_contribution.index_add_(0, row, edge_embeddings)
        edge_contribution.index_add_(0, col, edge_embeddings)
        x = x + edge_contribution
        
        # GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.post_mp(x)
        return x


class GraphSAGEStack(nn.Module):
    """
    GraphSAGE (Sample and Aggregate) stack for graph encoding.
    Uses neighborhood sampling and various aggregation functions.
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim, num_layers, 
                 dropout=0.0, aggr='mean'):
        super(GraphSAGEStack, self).__init__()
        
        assert num_layers >= 1, 'Number of layers must be >= 1'
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Edge feature projection
        self.edge_encoder = nn.Linear(edge_attr_dim, in_channels)
        
        # Build SAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(out_channels, out_channels, aggr=aggr))
        
        # Batch normalization
        self.bns = nn.ModuleList([
            BatchNorm1d(out_channels) 
            for _ in range(num_layers - 1)
        ])
        
        # Post-processing MLP
        self.post_mp = Sequential(
            Linear(out_channels, out_channels),
            nn.Dropout(dropout),
            Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through GraphSAGE stack."""
        # Encode edge features
        row, col = edge_index
        edge_embeddings = self.edge_encoder(edge_attr)
        
        edge_contribution = torch.zeros_like(x)
        edge_contribution.index_add_(0, row, edge_embeddings)
        edge_contribution.index_add_(0, col, edge_embeddings)
        x = x + edge_contribution
        
        # GraphSAGE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.post_mp(x)
        return x


class GINStack(nn.Module):
    """
    Graph Isomorphism Network (GIN) stack for graph encoding.
    Theoretically more powerful than GCN for graph representation.
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim, num_layers, 
                 dropout=0.0):
        super(GINStack, self).__init__()
        
        assert num_layers >= 1, 'Number of layers must be >= 1'
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Edge feature projection
        self.edge_encoder = nn.Linear(edge_attr_dim, in_channels)
        
        # Build GIN layers
        self.convs = nn.ModuleList()
        
        # First layer
        nn1 = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        self.convs.append(GINConv(nn1))
        
        # Remaining layers
        for _ in range(num_layers - 1):
            nn_i = Sequential(
                Linear(out_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels)
            )
            self.convs.append(GINConv(nn_i))
        
        # Batch normalization
        self.bns = nn.ModuleList([
            BatchNorm1d(out_channels) 
            for _ in range(num_layers - 1)
        ])
        
        # Post-processing MLP
        self.post_mp = Sequential(
            Linear(out_channels, out_channels),
            nn.Dropout(dropout),
            Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """Forward pass through GIN stack."""
        # Encode edge features
        row, col = edge_index
        edge_embeddings = self.edge_encoder(edge_attr)
        
        edge_contribution = torch.zeros_like(x)
        edge_contribution.index_add_(0, row, edge_embeddings)
        edge_contribution.index_add_(0, col, edge_embeddings)
        x = x + edge_contribution
        
        # GIN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.post_mp(x)
        return x


def create_gnn_stack(gnn_type, in_channels, out_channels, edge_attr_dim, 
                     num_layers, **kwargs):
    """
    Factory function to create different GNN architectures.
    
    Args:
        gnn_type: Type of GNN ('GAT', 'GCN', 'GraphSAGE', 'GIN')
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        edge_attr_dim: Edge feature dimension
        num_layers: Number of GNN layers
        **kwargs: Additional arguments specific to each GNN type
        
    Returns:
        GNN stack module
    """
    from .gat import GNNStack as GATStack
    
    if gnn_type.upper() == 'GAT':
        return GATStack(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_attr_dim=edge_attr_dim,
            num_layers=num_layers,
            **kwargs
        )
    elif gnn_type.upper() == 'GCN':
        return GCNStack(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_attr_dim=edge_attr_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.0)
        )
    elif gnn_type.upper() == 'GRAPHSAGE':
        return GraphSAGEStack(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_attr_dim=edge_attr_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.0),
            aggr=kwargs.get('aggr', 'mean')
        )
    elif gnn_type.upper() == 'GIN':
        return GINStack(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_attr_dim=edge_attr_dim,
            num_layers=num_layers,
            dropout=kwargs.get('dropout', 0.0)
        )
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}. Choose from 'GAT', 'GCN', 'GraphSAGE', 'GIN'")
