"""
Graph Attention Network (GAT) implementation with edge features
Based on the paper: Graph-Based Video-Language Learning with Multi-Grained Audio-Visual Alignment
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import Union, Tuple, Optional


class GAT(MessagePassing):
    """
    Graph Attention Network layer with edge features support.
    
    This implementation extends standard GAT to include edge features in the attention mechanism.
    """
    def __init__(self, 
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, 
                 edge_in_channels: int, 
                 heads: int = 1,
                 negative_slope: float = 0.2,
                 dropout: float = 0.0,
                 add_self_loops: bool = True, 
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear transformations for nodes
        if isinstance(in_channels, int):
            self.lin_l = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = nn.Linear(in_channels[0], heads * out_channels, bias=False)
            self.lin_r = nn.Linear(in_channels[1], heads * out_channels, bias=False)

        # Attention parameters for nodes
        self.att_l = Parameter(torch.zeros(heads, out_channels))
        self.att_r = Parameter(torch.zeros(heads, out_channels))
        
        # Linear transformation and attention for edges
        self.lin_e = nn.Linear(edge_in_channels, heads * out_channels, bias=False)
        self.att_e = Parameter(torch.zeros(heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.lin_e.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.att_e)

    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Forward pass of GAT layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_in_channels]
            size: Size of the graph (optional)
            
        Returns:
            Updated node features [num_nodes, heads * out_channels]
        """
        H, C = self.heads, self.out_channels

        # Transform node features
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        
        # Compute attention scores for source and target nodes
        alpha_l = self.att_l.unsqueeze(0) * x_l
        alpha_r = self.att_r.unsqueeze(0) * x_r

        # Transform edge features and compute attention scores
        e = self.lin_e(edge_attr).view(-1, H, C)
        alpha_e = self.att_e.unsqueeze(0) * e

        # Propagate messages
        out = self.propagate(
            edge_index=edge_index, 
            x=(x_l, x_r), 
            alpha=(alpha_l, alpha_r), 
            alpha_e=alpha_e, 
            size=size
        ).view(-1, H * C)

        return out

    def message(self, x_j, alpha_j, alpha_i, alpha_e, index, ptr, size_i):
        """
        Construct messages from source nodes to target nodes.
        
        The attention mechanism combines source node, target node, and edge features.
        """
        # Combine attention scores from source, target, and edge
        alpha = alpha_i + alpha_j + alpha_e
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight messages by attention scores
        out = alpha * x_j
        return out


class GNNStack(nn.Module):
    """
    Stack of GAT layers with batch normalization and residual connections.
    """
    def __init__(self, in_channels, out_channels, edge_attr_dim, num_layers, 
                 heads=4, dropout=0.0, negative_slope=0.2):
        super(GNNStack, self).__init__()
        
        assert num_layers >= 1, 'Number of layers must be >= 1'
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GAT(in_channels=in_channels, 
                out_channels=out_channels,
                edge_in_channels=edge_attr_dim, 
                heads=heads, 
                negative_slope=negative_slope, 
                dropout=dropout)
        )
        
        # Add remaining layers
        for l in range(num_layers - 1):
            self.convs.append(
                GAT(in_channels=heads * out_channels, 
                    out_channels=out_channels,
                    edge_in_channels=edge_attr_dim, 
                    heads=heads, 
                    negative_slope=negative_slope, 
                    dropout=dropout)
            )
        
        # Batch normalization for intermediate layers
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(heads * out_channels) 
            for _ in range(num_layers - 1)
        ])

        # Post-processing MLP
        self.post_mp = nn.Sequential(
            nn.Linear(heads * out_channels, out_channels), 
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass through the GNN stack.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_attr: Edge features
            batch: Batch assignment for nodes (optional)
            
        Returns:
            Updated node features
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            
            # Apply batch norm, activation, and dropout for intermediate layers
            if i != self.num_layers - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final post-processing
        x = self.post_mp(x)
        return x
