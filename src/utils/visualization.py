"""
Visualization utilities for AVQA-GNN model.
Includes attention visualization, graph visualization, and performance analysis.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import Dict, List
import os


class AttentionVisualizer:
    """Visualize attention weights in the model."""
    
    @staticmethod
    def visualize_cross_attention(attention_matrix, save_path=None):
        """
        Visualize cross-modal attention between audio and visual features.
        
        Args:
            attention_matrix: [B, T_v, T_a] attention matrix
            save_path: Path to save the figure
        """
        plt.figure(figsize=(10, 8))
        
        # Take first sample in batch
        if len(attention_matrix.shape) == 3:
            attention_matrix = attention_matrix[0]
        
        # Convert to numpy
        if torch.is_tensor(attention_matrix):
            attention_matrix = attention_matrix.cpu().detach().numpy()
        
        sns.heatmap(attention_matrix, cmap='viridis', cbar=True, 
                   xticklabels=5, yticklabels=5)
        plt.xlabel('Audio Features')
        plt.ylabel('Visual Features')
        plt.title('Cross-Modal Attention Weights')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def visualize_gat_attention(edge_index, edge_weights, num_nodes, 
                                node_labels=None, save_path=None):
        """
        Visualize GAT attention weights on the graph structure.
        
        Args:
            edge_index: [2, num_edges] edge connectivity
            edge_weights: [num_edges] attention weights
            num_nodes: Number of nodes
            node_labels: Labels for nodes (optional)
            save_path: Path to save the figure
        """
        # Create networkx graph
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # Convert to numpy
        if torch.is_tensor(edge_index):
            edge_index = edge_index.cpu().numpy()
        if torch.is_tensor(edge_weights):
            edge_weights = edge_weights.cpu().numpy()
        
        # Add edges with weights
        edges = [(edge_index[0, i], edge_index[1, i], 
                 {'weight': edge_weights[i]}) 
                for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        
        # Plot
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', 
                              alpha=0.9)
        
        # Draw edges with varying widths based on attention weights
        edge_weights_normalized = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights_normalized, 
                              alpha=0.6, edge_color='gray', 
                              arrows=True, arrowsize=20)
        
        # Draw labels
        if node_labels:
            labels = {i: node_labels[i] for i in range(num_nodes)}
        else:
            labels = {i: f'N{i}' for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('Graph Attention Network - Attention Weights')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph attention visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class GraphVisualizer:
    """Visualize scene graphs and query graphs."""
    
    @staticmethod
    def visualize_scene_graph(data, save_path=None):
        """
        Visualize a scene graph.
        
        Args:
            data: PyTorch Geometric Data object
            save_path: Path to save the figure
        """
        G = nx.DiGraph()
        
        # Get node labels
        node_labels = data.node_labels if hasattr(data, 'node_labels') else \
                     [f'N{i}' for i in range(data.x.shape[0])]
        
        # Add nodes
        for i, label in enumerate(node_labels):
            G.add_node(i, label=label)
        
        # Add edges
        edge_index = data.edge_index.cpu().numpy()
        edge_labels = data.edge_labels if hasattr(data, 'edge_labels') else \
                     [''] * edge_index.shape[1]
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst, label=edge_labels[i])
        
        # Plot
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes with labels
        nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightcoral', 
                              alpha=0.9)
        nx.draw_networkx_labels(G, pos, {i: node_labels[i] for i in range(len(node_labels))}, 
                               font_size=10, font_weight='bold')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, 
                              edge_color='gray', width=2)
        
        # Draw edge labels
        edge_label_dict = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_label_dict, font_size=8)
        
        plt.title('Scene Graph')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scene graph visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class PerformanceAnalyzer:
    """Analyze model performance."""
    
    @staticmethod
    def plot_training_curves(history, save_path=None):
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def compare_gnn_architectures(results_dict, save_path=None):
        """
        Compare performance of different GNN architectures.
        
        Args:
            results_dict: Dictionary mapping GNN type to accuracy
            save_path: Path to save the figure
        """
        architectures = list(results_dict.keys())
        accuracies = list(results_dict.values())
        
        if not accuracies:
            print("Warning: No trained models found. Please train models first using:")
            print("  python train.py --gnn_type GAT --exp_name avqa_gnn_gat")
            print("  python train.py --gnn_type GCN --exp_name avqa_gnn_gcn")
            print("  python train.py --gnn_type GraphSAGE --exp_name avqa_gnn_graphsage")
            print("  python train.py --gnn_type GIN --exp_name avqa_gnn_gin")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(architectures, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('GNN Architecture', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Comparison of GNN Architectures on AVQA Task', fontsize=14, fontweight='bold')
        plt.ylim(0, max(accuracies) * 1.2)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Architecture comparison saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: [num_classes, num_classes] confusion matrix
            class_names: List of class names
            save_path: Path to save the figure
        """
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def analyze_question_types(results_by_type, save_path=None):
        """
        Analyze performance by question type.
        
        Args:
            results_by_type: Dictionary mapping question type to accuracy
            save_path: Path to save the figure
        """
        question_types = list(results_by_type.keys())
        accuracies = list(results_by_type.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(question_types, accuracies, color='steelblue')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{acc:.2f}%',
                    ha='left', va='center', fontweight='bold')
        
        plt.xlabel('Accuracy (%)', fontsize=12)
        plt.ylabel('Question Type', fontsize=12)
        plt.title('Performance by Question Type', fontsize=14, fontweight='bold')
        plt.xlim(0, max(accuracies) * 1.2)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Question type analysis saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def create_visualization_examples():
    """Create example visualizations for documentation."""
    vis_dir = './visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Example 1: Cross-attention
    attention_matrix = torch.softmax(torch.randn(10, 10), dim=-1)
    AttentionVisualizer.visualize_cross_attention(
        attention_matrix, 
        save_path=os.path.join(vis_dir, 'cross_attention_example.png')
    )
    
    # Example 2: Training curves
    history = {
        'train_loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5, 0.4],
        'train_acc': [30, 40, 50, 58, 65, 70, 75, 78, 80, 82],
        'val_loss': [2.6, 2.1, 1.6, 1.3, 1.1, 0.9, 0.8, 0.7, 0.65, 0.6],
        'val_acc': [28, 38, 48, 55, 62, 67, 72, 74, 76, 78]
    }
    PerformanceAnalyzer.plot_training_curves(
        history,
        save_path=os.path.join(vis_dir, 'training_curves_example.png')
    )
    
    # Example 3: GNN comparison
    results = {
        'GAT': 78.5,
        'GCN': 72.3,
        'GraphSAGE': 75.1,
        'GIN': 76.8
    }
    PerformanceAnalyzer.compare_gnn_architectures(
        results,
        save_path=os.path.join(vis_dir, 'gnn_comparison_example.png')
    )
    
    # Example 4: Question type analysis
    question_results = {
        'Existential': 82.5,
        'Location': 75.3,
        'Counting': 68.7,
        'Comparative': 71.2,
        'Temporal': 79.8
    }
    PerformanceAnalyzer.analyze_question_types(
        question_results,
        save_path=os.path.join(vis_dir, 'question_types_example.png')
    )
    
    print("Example visualizations created successfully!")


if __name__ == '__main__':
    create_visualization_examples()
