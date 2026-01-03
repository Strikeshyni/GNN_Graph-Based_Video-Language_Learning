"""
Training script for AVQA-GNN model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from src.models.avqa_gnn import AVQA_GNN
from src.models.alternative_gnns import create_gnn_stack
from src.data.dataset import create_dataloaders


class Trainer:
    """Trainer class for AVQA-GNN model."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dataloaders
        print("Loading dataset...")
        self.train_loader, self.val_loader, self.test_loader, self.idx_to_answer = \
            create_dataloaders(args.data_dir, args.batch_size, args.num_workers)
        
        # Update num_classes based on dataset
        args.num_classes = len(self.idx_to_answer)
        
        # Create model
        print(f"Creating model with GNN type: {args.gnn_type}")
        self.model = AVQA_GNN(args).to(self.device)
        
        # If using alternative GNN, replace the GNN stacks
        if args.gnn_type != 'GAT':
            self.model.scene_gnn = create_gnn_stack(
                args.gnn_type,
                in_channels=args.node_dim,
                out_channels=args.out_channels,
                edge_attr_dim=args.edge_dim,
                num_layers=args.gnn_layers,
                dropout=args.dropout
            ).to(self.device)
            
            self.model.query_gnn = create_gnn_stack(
                args.gnn_type,
                in_channels=args.node_dim,
                out_channels=args.out_channels,
                edge_attr_dim=args.edge_dim,
                num_layers=args.gnn_layers,
                dropout=args.dropout
            ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f'{args.exp_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'))
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            audio_feat = batch['audio_feat'].to(self.device)
            visual_feat = batch['visual_feat'].to(self.device)
            question_feat = batch['question_feat'].to(self.device)
            answers = batch['answer'].to(self.device)
            
            # Move graph data to device
            sg_x, sg_edge_index, sg_edge_attr, sg_batch = batch['sg_data']
            sg_data = (sg_x.to(self.device), sg_edge_index.to(self.device), 
                      sg_edge_attr.to(self.device), sg_batch.to(self.device))
            
            qg_x, qg_edge_index, qg_edge_attr, qg_batch = batch['qg_data']
            qg_data = (qg_x.to(self.device), qg_edge_index.to(self.device),
                      qg_edge_attr.to(self.device), qg_batch.to(self.device))
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(audio_feat, visual_feat, question_feat, sg_data, qg_data)
            loss = self.criterion(logits, answers)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == answers).sum().item()
            total += answers.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                audio_feat = batch['audio_feat'].to(self.device)
                visual_feat = batch['visual_feat'].to(self.device)
                question_feat = batch['question_feat'].to(self.device)
                answers = batch['answer'].to(self.device)
                
                # Move graph data to device
                sg_x, sg_edge_index, sg_edge_attr, sg_batch = batch['sg_data']
                sg_data = (sg_x.to(self.device), sg_edge_index.to(self.device),
                          sg_edge_attr.to(self.device), sg_batch.to(self.device))
                
                qg_x, qg_edge_index, qg_edge_attr, qg_batch = batch['qg_data']
                qg_data = (qg_x.to(self.device), qg_edge_index.to(self.device),
                          qg_edge_attr.to(self.device), qg_batch.to(self.device))
                
                # Forward pass
                logits = self.model(audio_feat, visual_feat, question_feat, sg_data, qg_data)
                loss = self.criterion(logits, answers)
                
                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == answers).sum().item()
                total += answers.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best validation accuracy: {val_acc:.2f}%")
            
            # Save regular checkpoint
            if epoch % self.args.save_freq == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Save final model and history
        self.save_checkpoint(self.args.epochs, is_best=False)
        self.save_history()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'args': self.args
        }
        
        if is_best:
            path = os.path.join(self.args.checkpoint_dir, f'{self.args.exp_name}_best.pth')
        else:
            path = os.path.join(self.args.checkpoint_dir, f'{self.args.exp_name}_epoch{epoch}.pth')
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def save_history(self):
        """Save training history."""
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        history_path = os.path.join(self.args.checkpoint_dir, f'{self.args.exp_name}_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train AVQA-GNN model')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default='./data/MUSIC-AVQA', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (use 0 to avoid multiprocessing issues)')
    
    # Model
    parser.add_argument('--gnn_type', type=str, default='GAT', 
                       choices=['GAT', 'GCN', 'GraphSAGE', 'GIN'],
                       help='Type of GNN to use')
    parser.add_argument('--node_dim', type=int, default=512, help='Node feature dimension')
    parser.add_argument('--edge_dim', type=int, default=512, help='Edge feature dimension')
    parser.add_argument('--out_channels', type=int, default=512, help='Output channels')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--gnn_heads', type=int, default=4, help='Number of attention heads (for GAT)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--lr_step', type=int, default=10, help='Learning rate decay step')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate decay gamma')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    
    # Logging
    parser.add_argument('--exp_name', type=str, default='avqa_gnn', help='Experiment name')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--save_freq', type=int, default=25, help='Save checkpoint frequency (epochs)')
    
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
