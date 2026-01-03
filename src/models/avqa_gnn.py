"""
Complete AVQA-GNN model implementation for Audio-Visual Question Answering
Based on: Graph-Based Video-Language Learning with Multi-Grained Audio-Visual Alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Bilinear, Parameter
from .gat import GNNStack


class MultiGrainedAlignment(nn.Module):
    """
    Multi-Grained Alignment module (MgA) for audio-visual feature fusion.
    Uses 1D convolutions with different kernel sizes to capture multi-scale patterns.
    """
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(MultiGrainedAlignment, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)

    def forward(self, input):
        """
        Apply multi-scale convolutions.
        
        Args:
            input: [B, C, T] - Batch, Channels, Time
            
        Returns:
            Tuple of 3 feature maps at different scales
        """
        conv_1 = self.conv1(input).permute(0, 2, 1)  # [B, T, C]
        conv_2 = self.conv2(input).permute(0, 2, 1)  # [B, T, C] (same size due to padding)
        conv_3 = self.conv3(input).permute(0, 2, 1)  # [B, T, C] (same size due to padding)

        return conv_1, conv_2, conv_3


class CrossAttention(nn.Module):
    """
    Cross-modal attention mechanism between audio and visual features.
    """
    def __init__(self, dim=512):
        super(CrossAttention, self).__init__()
        self.w = Linear(dim, dim)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, audio_conv_list, video_conv_list):
        """
        Compute cross-attention between audio and visual features at multiple scales.
        
        Args:
            audio_conv_list: List of 3 audio feature tensors [B, T, C]
            video_conv_list: List of 3 video feature tensors [B, T, C]
            
        Returns:
            f_v: List of attended visual features for each scale combination
            f_a: List of attended audio features for each scale combination
        """
        f_v = [[], [], []]  # Visual features attended by audio
        f_a = [[], [], []]  # Audio features attended by visual
        
        for i, video_conv in enumerate(video_conv_list):
            for j, audio_conv in enumerate(audio_conv_list):
                # Compute attention scores
                scores = torch.bmm(self.w(video_conv), audio_conv.permute(0, 2, 1))
                scores = scores / torch.sqrt(torch.tensor(video_conv.shape[-1], dtype=torch.float32))
                a_ij = F.softmax(scores, dim=-1)  # [B, T_v, T_a]
                
                # Apply attention
                f_v[i].append(torch.bmm(a_ij, audio_conv))  # Video attended by audio
                f_a[j].append(torch.bmm(a_ij.permute(0, 2, 1), video_conv))  # Audio attended by video

        return f_v, f_a


class HierarchicalMatch(nn.Module):
    """
    Hierarchical matching module to combine multi-scale features with graph-based representations.
    """
    def __init__(self, N=3, dim=512):
        super(HierarchicalMatch, self).__init__()
        self.N = N
        self.dim = dim

    def forward(self, joint, f, B, device='cuda'):
        """
        Hierarchically match and combine multi-scale features.
        
        Args:
            joint: Joint graph representation [B, dim]
            f: List of multi-scale features (3x3 grid)
            B: Batch size
            device: Device to use
            
        Returns:
            Combined feature representation [B, dim]
        """
        N = self.N
        device = joint.device
        
        # First level: compute weights for each scale combination
        b = torch.zeros(N, N, B, device=device)
        for i, f_i in enumerate(f):
            for j, f_ij in enumerate(f_i):
                f_ij_tmp = torch.mean(f_ij, dim=1)  # [B, dim]
                numerator = torch.bmm(joint.unsqueeze(1), f_ij_tmp.unsqueeze(-1)).squeeze()
                
                # Compute denominator with numerical stability
                denominator_list = []
                for r in range(N):
                    val = torch.bmm(joint.unsqueeze(1), f[i][r].mean(dim=1).unsqueeze(-1)).squeeze()
                    denominator_list.append(val)
                denominator = torch.stack(denominator_list).sum(dim=0) + 1e-8
                
                b[i][j] = numerator / denominator

        # Second level: combine within same kernel size
        f_ii = []
        for i, f_i in enumerate(f):
            weighted = torch.stack([b[i][j][:, None, None] * f_ij for j, f_ij in enumerate(f_i)])
            f_ii.append(weighted.sum(dim=0))

        # Third level: combine across kernel sizes
        lambda_i = torch.zeros(N, B, device=device)
        for i, f_i in enumerate(f_ii):
            f_i_tmp = torch.mean(f_i, dim=1)  # [B, dim]
            numerator = torch.bmm(joint.unsqueeze(1), f_i_tmp.unsqueeze(-1)).squeeze()
            
            denominator_list = []
            for r in range(N):
                val = torch.bmm(joint.unsqueeze(1), f_ii[r].mean(dim=1).unsqueeze(-1)).squeeze()
                denominator_list.append(val)
            denominator = torch.stack(denominator_list).sum(dim=0) + 1e-8
            
            lambda_i[i] = numerator / denominator

        # Final weighted combination
        result = torch.stack([lambda_i[i][:, None] * f_i.mean(dim=1) for i, f_i in enumerate(f_ii)])
        return result.sum(dim=0)  # [B, dim]


class AVQA_GNN(nn.Module):
    """
    Complete Audio-Visual Question Answering model with Graph Neural Networks.
    
    Architecture:
    1. Encode scene graphs and query graphs with separate GNNs
    2. Create joint graph-based representation
    3. Multi-grained alignment of audio-visual features
    4. Hierarchical matching with graph representations
    5. Final classification
    """
    def __init__(self, args):
        super(AVQA_GNN, self).__init__()
        
        # Hyperparameters
        self.num_classes = getattr(args, 'num_classes', 42)
        self.node_dim = getattr(args, 'node_dim', 512)
        self.edge_dim = getattr(args, 'edge_dim', 512)
        self.out_channels = getattr(args, 'out_channels', 512)
        self.gnn_layers = getattr(args, 'gnn_layers', 2)
        self.gnn_heads = getattr(args, 'gnn_heads', 4)
        self.dropout = getattr(args, 'dropout', 0.1)
        
        # Graph encoders for scene graphs and query graphs
        self.scene_gnn = GNNStack(
            in_channels=self.node_dim,
            out_channels=self.out_channels,
            edge_attr_dim=self.edge_dim,
            num_layers=self.gnn_layers,
            heads=self.gnn_heads,
            dropout=self.dropout
        )
        
        self.query_gnn = GNNStack(
            in_channels=self.node_dim,
            out_channels=self.out_channels,
            edge_attr_dim=self.edge_dim,
            num_layers=self.gnn_layers,
            heads=self.gnn_heads,
            dropout=self.dropout
        )
        
        # Joint representation modules
        self.wv = Linear(self.out_channels, self.out_channels)
        self.joint_linear = Bilinear(self.out_channels, self.out_channels, self.out_channels)
        
        # Audio encoding
        self.lin_a = Sequential(
            Linear(128, 512),  # VGGish features are 128-dim
            ReLU(),
            Linear(512, self.out_channels)
        )
        
        # Visual encoding  
        self.lin_v = Sequential(
            Linear(512, 512),  # CLIP features are 512-dim
            ReLU(),
            Linear(512, self.out_channels)
        )
        
        # Question encoding
        self.lin_q = Sequential(
            Linear(512, 512),
            ReLU(),
            Linear(512, self.out_channels)
        )
        
        # Multi-grained alignment
        self.mga_v = MultiGrainedAlignment(self.out_channels, self.out_channels)
        self.mga_a = MultiGrainedAlignment(self.out_channels, self.out_channels)
        self.cross_attn = CrossAttention(self.out_channels)
        
        # Hierarchical matching
        self.match_v = HierarchicalMatch(N=3, dim=self.out_channels)
        self.match_a = HierarchicalMatch(N=3, dim=self.out_channels)
        
        # Final fusion and classification
        self.tanh_avq = nn.Tanh()
        self.classifier = Sequential(
            Linear(self.out_channels, self.out_channels),
            ReLU(),
            nn.Dropout(self.dropout),
            Linear(self.out_channels, self.num_classes)
        )

    def forward(self, audio_feat, visual_feat, question_feat, sg_data, qg_data):
        """
        Forward pass of the AVQA-GNN model.
        
        Args:
            audio_feat: Audio features [B, T, 128]
            visual_feat: Visual features [B, T, 512]
            question_feat: Question features [B, 1, 512]
            sg_data: Scene graph data (x, edge_index, edge_attr, batch)
            qg_data: Query graph data (x, edge_index, edge_attr, batch)
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        B = audio_feat.shape[0]
        device = audio_feat.device
        
        # 1. Encode graphs
        sg_x, sg_edge_index, sg_edge_attr, sg_batch = sg_data
        qg_x, qg_edge_index, qg_edge_attr, qg_batch = qg_data
        
        video_graph = self.scene_gnn(sg_x, sg_edge_index, sg_edge_attr, sg_batch)  # [N_v, 512]
        query_graph = self.query_gnn(qg_x, qg_edge_index, qg_edge_attr, qg_batch)  # [N_q, 512]
        
        # Aggregate graph features per sample in batch using global mean pooling
        # This handles variable number of nodes per graph
        from torch_geometric.nn import global_mean_pool
        video = global_mean_pool(video_graph, sg_batch)  # [B, 512]
        query = global_mean_pool(query_graph, qg_batch)  # [B, 512]
        
        # Expand to match expected dimensions for similarity computation
        video = video.unsqueeze(1)  # [B, 1, 512]
        query = query.unsqueeze(1)  # [B, 1, 512]
        
        # 2. Joint graph representation
        # Compute similarity between video and query graph embeddings
        sim = torch.bmm(video, query.permute(0, 2, 1))  # [B, 1, 1]
        temperature = 1.0
        
        # Video-query joint representation
        v_joint = self.wv(video.squeeze(1))  # [B, 512]
        q_joint = query.squeeze(1)  # [B, 512]
        
        # Combined joint representation
        vq_joint = self.joint_linear(v_joint, q_joint)  # [B, 512]
        
        # 3. Encode audio-visual features
        audio_feat = self.lin_a(audio_feat)  # [B, T, 512]
        visual_feat = self.lin_v(visual_feat)  # [B, T, 512]
        question_feat = self.lin_q(question_feat)  # [B, 1, 512]
        
        # 4. Multi-grained alignment
        audio_conv_1, audio_conv_2, audio_conv_3 = self.mga_a(audio_feat.permute(0, 2, 1))
        video_conv_1, video_conv_2, video_conv_3 = self.mga_v(visual_feat.permute(0, 2, 1))
        
        audio_conv_list = [audio_conv_1, audio_conv_2, audio_conv_3]
        video_conv_list = [video_conv_1, video_conv_2, video_conv_3]
        
        f_v, f_a = self.cross_attn(audio_conv_list, video_conv_list)
        
        # 5. Hierarchical matching
        f_v_matched = self.match_v(vq_joint, f_v, B, device)  # [B, 512]
        f_a_matched = self.match_a(vq_joint, f_a, B, device)  # [B, 512]
        
        # 6. Gated fusion
        z_v = torch.sigmoid(vq_joint * f_v_matched)  # [B, 512]
        z_a = torch.sigmoid(vq_joint * f_a_matched)  # [B, 512]
        
        f_m = z_v * f_v_matched + z_a * f_a_matched  # [B, 512]
        
        # 7. Final feature combination
        avq_feat = f_m * question_feat.squeeze(1)  # [B, 512]
        avq_feat = self.tanh_avq(avq_feat)
        
        # 8. Classification
        logits = self.classifier(avq_feat)  # [B, num_classes]
        
        return logits
