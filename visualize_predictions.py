"""
Visual Testing Script for AVQA-GNN Models
Shows video frames, question, predicted answer, and ground truth for better understanding
"""

import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.models.avqa_gnn import AVQA_GNN
from src.models.alternative_gnns import create_gnn_stack
from src.data.dataset import create_dataloaders


class VisualTester:
    """Visual tester for AVQA-GNN models with frame visualization."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        print("Loading dataset...")
        _, _, self.test_loader, self.idx_to_answer, _ = \
            create_dataloaders(args.data_dir, args.batch_size, args.num_workers)
        
        self.answer_to_idx = {v: k for k, v in self.idx_to_answer.items()}
        self.num_classes = len(self.idx_to_answer)
        
        # Load annotations to get questions
        self.annotations = self.load_annotations(args.data_dir)
        
        # Load model
        self.model = self.load_model(args.checkpoint, args.gnn_type)
        self.model.eval()
    
    def load_annotations(self, data_dir):
        """Load test annotations to get questions with template values filled."""
        anno_path = os.path.join(data_dir, 'test_annotations.json')
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                annotations = json.load(f)
            # Create a mapping from video_id to question
            video_to_question = {}
            for anno in annotations:
                video_id = anno.get('video_id', '')
                question_raw = anno.get('question', anno.get('question_content', 'Question not available'))
                
                # Fill template values (e.g., <Object>, <LR>, <FL>, <LRer>)
                templ_values_str = anno.get('templ_values', '[]')
                try:
                    import ast
                    templ_values = ast.literal_eval(templ_values_str) if isinstance(templ_values_str, str) else templ_values_str
                except:
                    templ_values = []
                
                if templ_values and '<' in question_raw:
                    # Split question into words
                    words = question_raw.rstrip().split(' ')
                    # Remove trailing punctuation from last word
                    if words and words[-1] and words[-1][-1] in '?？':
                        words[-1] = words[-1][:-1]
                    
                    # Replace placeholders with template values
                    p = 0
                    for i, word in enumerate(words):
                        if '<' in word and p < len(templ_values):
                            words[i] = templ_values[p]
                            p += 1
                    
                    question_filled = ' '.join(words).strip()
                    if not question_filled.endswith('?'):
                        question_filled += '?'
                    video_to_question[video_id] = question_filled
                else:
                    video_to_question[video_id] = question_raw
            return video_to_question
        return {}
    
    def load_model(self, checkpoint_path, gnn_type='GAT'):
        """Load a trained model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Creating a new model for testing (untrained)...")
            # Create args for model
            class ModelArgs:
                def __init__(self):
                    self.node_dim = 512
                    self.edge_dim = 512
                    self.out_channels = 512
                    self.gnn_layers = 2
                    self.gnn_heads = 4
                    self.dropout = 0.1
                    self.num_classes = len(self.idx_to_answer)
            
            model_args = ModelArgs()
            model_args.num_classes = self.num_classes
            model = AVQA_GNN(model_args).to(self.device)
            return model
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'args' in checkpoint:
            model_args = checkpoint['args']
        else:
            # Fallback to default args
            class ModelArgs:
                def __init__(self):
                    self.node_dim = 512
                    self.edge_dim = 512
                    self.out_channels = 512
                    self.gnn_layers = 2
                    self.gnn_heads = 4
                    self.dropout = 0.1
                    self.num_classes = len(self.idx_to_answer)
            
            model_args = ModelArgs()
            model_args.num_classes = self.num_classes
        
        model = AVQA_GNN(model_args).to(self.device)
        
        # Replace GNN if needed
        if gnn_type != 'GAT':
            model.scene_gnn = create_gnn_stack(
                gnn_type,
                in_channels=model_args.node_dim,
                out_channels=model_args.out_channels,
                edge_attr_dim=model_args.edge_dim,
                num_layers=model_args.gnn_layers,
                dropout=model_args.dropout
            ).to(self.device)
            
            model.query_gnn = create_gnn_stack(
                gnn_type,
                in_channels=model_args.node_dim,
                out_channels=model_args.out_channels,
                edge_attr_dim=model_args.edge_dim,
                num_layers=model_args.gnn_layers,
                dropout=model_args.dropout
            ).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def load_video_frames(self, video_id):
        """
        Load video frames for visualization.
        
        Args:
            video_id: Video identifier
            
        Returns:
            List of PIL Images or None if frames not available
        """
        frames_dir = os.path.join(self.args.data_dir, 'frames', video_id)
        
        if not os.path.exists(frames_dir):
            # Return placeholder frames
            return None
        
        # Load a few representative frames (e.g., 5 frames evenly spaced)
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
        
        if len(frame_files) == 0:
            return None
        
        # Select 5 evenly spaced frames
        num_frames = min(5, len(frame_files))
        indices = np.linspace(0, len(frame_files)-1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame_path = os.path.join(frames_dir, frame_files[idx])
            try:
                img = Image.open(frame_path).convert('RGB')
                frames.append(img)
            except:
                continue
        
        return frames if len(frames) > 0 else None
    
    def visualize_prediction(self, video_id, question, predicted_answer, ground_truth, 
                           confidence, is_correct, save_path=None):
        """
        Create a visualization of the prediction.
        
        Args:
            video_id: Video identifier
            question: Question text
            predicted_answer: Predicted answer
            ground_truth: Ground truth answer
            confidence: Prediction confidence
            is_correct: Whether prediction is correct
            save_path: Path to save the visualization
        """
        # Load frames
        frames = self.load_video_frames(video_id)
        
        # Create figure
        if frames:
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 5, hspace=0.3, wspace=0.3)
            
            # Display frames
            for i, frame in enumerate(frames):
                ax = fig.add_subplot(gs[0, i])
                ax.imshow(frame)
                ax.axis('off')
                ax.set_title(f'Frame {i+1}', fontsize=10)
            
            # Text information
            ax_text = fig.add_subplot(gs[1, :])
            ax_text.axis('off')
        else:
            fig, ax_text = plt.subplots(1, 1, figsize=(16, 6))
            ax_text.axis('off')
        
        # Prepare text
        status_color = 'green' if is_correct else 'red'
        status_symbol = '✓' if is_correct else '✗'
        
        info_text = f"""
Video ID: {video_id}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Question:
  {question}

Predicted Answer:  {predicted_answer}  (confidence: {confidence:.2%})

Ground Truth:      {ground_truth}

Result: {status_symbol} {'CORRECT' if is_correct else 'INCORRECT'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        ax_text.text(0.05, 0.5, info_text, fontsize=14, verticalalignment='center',
                    family='monospace', color=status_color if not is_correct else 'black')
        
        # Add a colored border around the prediction result
        if is_correct:
            rect = patches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=5, 
                                    edgecolor='green', facecolor='none', 
                                    transform=fig.transFigure)
        else:
            rect = patches.Rectangle((0.02, 0.02), 0.96, 0.96, linewidth=5, 
                                    edgecolor='red', facecolor='none', 
                                    transform=fig.transFigure)
        fig.patches.append(rect)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved visualization to {save_path}")
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()
    
    def test_with_visualization(self, num_samples=5):
        """
        Test the model and create visualizations for a few samples.
        
        Args:
            num_samples: Number of samples to visualize
        """
        print(f"\n{'='*60}")
        print(f"Visual Testing: {self.args.gnn_type} Model")
        print(f"{'='*60}\n")
        
        os.makedirs('./visualizations/predictions', exist_ok=True)
        
        self.model.eval()
        samples_visualized = 0
        correct_count = 0
        total_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                audio_feat = batch['audio_feat'].to(self.device)
                visual_feat = batch['visual_feat'].to(self.device)
                question_feat = batch['question_feat'].to(self.device)
                
                # Move graph data to device
                sg_x, sg_edge_index, sg_edge_attr, sg_batch = batch['sg_data']
                sg_data = (
                    sg_x.to(self.device),
                    sg_edge_index.to(self.device),
                    sg_edge_attr.to(self.device),
                    sg_batch.to(self.device)
                )
                
                qg_x, qg_edge_index, qg_edge_attr, qg_batch = batch['qg_data']
                qg_data = (
                    qg_x.to(self.device),
                    qg_edge_index.to(self.device),
                    qg_edge_attr.to(self.device),
                    qg_batch.to(self.device)
                )
                
                answers = batch['answer'].to(self.device)
                video_ids = batch['video_ids']
                questions = batch.get('questions', [None] * len(video_ids))  # Get questions from batch
                
                # Forward pass
                logits = self.model(audio_feat, visual_feat, question_feat, sg_data, qg_data)
                probs = torch.softmax(logits, dim=1)
                predicted = torch.argmax(logits, dim=1)
                
                # Process each sample in batch
                for i in range(len(video_ids)):
                    video_id = video_ids[i]
                    pred_idx = predicted[i].item()
                    gt_idx = answers[i].item()
                    confidence = probs[i, pred_idx].item()
                    
                    pred_answer = self.idx_to_answer[pred_idx]
                    gt_answer = self.idx_to_answer[gt_idx]
                    is_correct = pred_idx == gt_idx
                    
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    
                    # Visualize first N samples
                    if samples_visualized < num_samples:
                        # Get question text directly from batch (correct for each sample)
                        question = questions[i] if questions[i] else f"Question about video {video_id}"
                        
                        save_path = f'./visualizations/predictions/sample_{samples_visualized+1}_{video_id}.png'
                        
                        print(f"\n{'─'*60}")
                        print(f"Sample {samples_visualized + 1}/{num_samples}")
                        print(f"{'─'*60}")
                        print(f"Video ID: {video_id}")
                        print(f"Question: {question}")
                        print(f"Predicted: {pred_answer} (confidence: {confidence:.2%})")
                        print(f"Ground Truth: {gt_answer}")
                        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                        
                        self.visualize_prediction(
                            video_id=video_id,
                            question=question,
                            predicted_answer=pred_answer,
                            ground_truth=gt_answer,
                            confidence=confidence,
                            is_correct=is_correct,
                            save_path=save_path
                        )
                        
                        samples_visualized += 1
                
                if samples_visualized >= num_samples:
                    break
        
        # Print overall statistics
        accuracy = 100.0 * correct_count / total_count if total_count > 0 else 0
        print(f"\n{'='*60}")
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
        print(f"{'='*60}\n")
        
        return accuracy


def main():
    parser = argparse.ArgumentParser(description='Visual Testing for AVQA-GNN')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default='./data/MUSIC-AVQA',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of workers')
    
    # Model
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/avqa_gnn_gat_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--gnn_type', type=str, default='GAT',
                       choices=['GAT', 'GCN', 'GraphSAGE', 'GIN'],
                       help='Type of GNN')
    
    # Visualization
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create tester
    tester = VisualTester(args)
    
    # Run visual testing
    tester.test_with_visualization(num_samples=args.num_samples)
    
    print("\nVisual testing completed!")
    print(f"Visualizations saved in: ./visualizations/predictions/")


if __name__ == '__main__':
    main()
