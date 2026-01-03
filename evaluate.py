"""
Evaluation and comparison script for AVQA-GNN models.
"""

import os
import torch
import argparse
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from src.models.avqa_gnn import AVQA_GNN
from src.models.alternative_gnns import create_gnn_stack
from src.data.dataset import create_dataloaders
from src.utils.visualization import PerformanceAnalyzer


class Evaluator:
    """Evaluator for AVQA-GNN models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        print("Loading dataset...")
        _, _, self.test_loader, self.idx_to_answer = \
            create_dataloaders(args.data_dir, args.batch_size, args.num_workers)
        
        self.num_classes = len(self.idx_to_answer)
        
    def load_model(self, checkpoint_path, gnn_type='GAT'):
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        args = checkpoint['args']
        
        # Create model
        model = AVQA_GNN(args).to(self.device)
        
        # Replace GNN if needed
        if gnn_type != 'GAT':
            model.scene_gnn = create_gnn_stack(
                gnn_type,
                in_channels=args.node_dim,
                out_channels=args.out_channels,
                edge_attr_dim=args.edge_dim,
                num_layers=args.gnn_layers,
                dropout=args.dropout
            ).to(self.device)
            
            model.query_gnn = create_gnn_stack(
                gnn_type,
                in_channels=args.node_dim,
                out_channels=args.out_channels,
                edge_attr_dim=args.edge_dim,
                num_layers=args.gnn_layers,
                dropout=args.dropout
            ).to(self.device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def evaluate(self, model):
        """Evaluate a model on the test set."""
        all_preds = []
        all_labels = []
        all_video_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluating'):
                # Move data to device
                audio_feat = batch['audio_feat'].to(self.device)
                visual_feat = batch['visual_feat'].to(self.device)
                question_feat = batch['question_feat'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                # Move graph data
                sg_x, sg_edge_index, sg_edge_attr, sg_batch = batch['sg_data']
                sg_data = (sg_x.to(self.device), sg_edge_index.to(self.device),
                          sg_edge_attr.to(self.device), sg_batch.to(self.device))
                
                qg_x, qg_edge_index, qg_edge_attr, qg_batch = batch['qg_data']
                qg_data = (qg_x.to(self.device), qg_edge_index.to(self.device),
                          qg_edge_attr.to(self.device), qg_batch.to(self.device))
                
                # Forward pass
                logits = model(audio_feat, visual_feat, question_feat, sg_data, qg_data)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(answers.cpu().numpy())
                all_video_ids.extend(batch['video_ids'])
        
        # Compute metrics
        accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        target_names = [self.idx_to_answer[i] for i in range(self.num_classes)]
        report = classification_report(all_labels, all_preds, target_names=target_names, 
                                      output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'labels': all_labels,
            'video_ids': all_video_ids
        }
    
    def compare_architectures(self, checkpoint_dir):
        """Compare different GNN architectures."""
        results = {}
        
        gnn_types = ['GAT', 'GCN', 'GraphSAGE', 'GIN']
        
        for gnn_type in gnn_types:
            checkpoint_path = os.path.join(checkpoint_dir, f'avqa_gnn_{gnn_type.lower()}_best.pth')
            
            if not os.path.exists(checkpoint_path):
                print(f"Checkpoint not found for {gnn_type}: {checkpoint_path}")
                continue
            
            print(f"\nEvaluating {gnn_type}...")
            model = self.load_model(checkpoint_path, gnn_type)
            result = self.evaluate(model)
            
            results[gnn_type] = {
                'accuracy': result['accuracy'],
                'report': result['classification_report']
            }
            
            print(f"{gnn_type} Accuracy: {result['accuracy']:.2f}%")
        
        return results
    
    def ablation_study(self, checkpoint_path):
        """Perform ablation study on model components."""
        model = self.load_model(checkpoint_path)
        
        results = {}
        
        # Full model
        print("\n1. Full model")
        result = self.evaluate(model)
        results['full_model'] = result['accuracy']
        print(f"Accuracy: {result['accuracy']:.2f}%")
        
        # Without multi-grained alignment (set to identity)
        print("\n2. Without multi-grained alignment")
        # This would require modifying the model architecture
        # For now, we just report the full model result
        
        # Without graph encoding
        print("\n3. Without graph encoding")
        # This would require modifying the forward pass
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate AVQA-GNN models')
    
    parser.add_argument('--data_dir', type=str, default='./data/MUSIC-AVQA')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_path', type=str, help='Path to specific checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--compare_architectures', action='store_true',
                       help='Compare different GNN architectures')
    parser.add_argument('--ablation', action='store_true',
                       help='Perform ablation study')
    parser.add_argument('--output_dir', type=str, default='./results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = Evaluator(args)
    
    if args.compare_architectures:
        print("Comparing GNN architectures...")
        results = evaluator.compare_architectures(args.checkpoint_dir)
        
        # Save results
        with open(os.path.join(args.output_dir, 'architecture_comparison.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Visualize comparison
        accuracies = {k: v['accuracy'] for k, v in results.items()}
        PerformanceAnalyzer.compare_gnn_architectures(
            accuracies,
            save_path=os.path.join(args.output_dir, 'architecture_comparison.png')
        )
        
    elif args.ablation:
        print("Performing ablation study...")
        if not args.checkpoint_path:
            args.checkpoint_path = os.path.join(args.checkpoint_dir, 'avqa_gnn_best.pth')
        
        results = evaluator.ablation_study(args.checkpoint_path)
        
        with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    else:
        print("Evaluating single model...")
        if not args.checkpoint_path:
            args.checkpoint_path = os.path.join(args.checkpoint_dir, 'avqa_gnn_best.pth')
        
        model = evaluator.load_model(args.checkpoint_path)
        results = evaluator.evaluate(model)
        
        print(f"\nTest Accuracy: {results['accuracy']:.2f}%")
        
        # Save results
        results_to_save = {
            'accuracy': results['accuracy'],
            'classification_report': results['classification_report']
        }
        
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Visualize confusion matrix
        PerformanceAnalyzer.plot_confusion_matrix(
            results['confusion_matrix'],
            [evaluator.idx_to_answer[i] for i in range(evaluator.num_classes)],
            save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
