"""
Dataset loader for MUSIC-AVQA dataset.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
from .scene_graph_parser import SceneGraphParser, QueryGraphParser


class AVQA_Dataset(Dataset):
    """
    Dataset for Audio-Visual Question Answering.
    
    The MUSIC-AVQA dataset contains:
    - Video frames with audio
    - Questions about the audio-visual content
    - Multiple choice answers
    """
    def __init__(self, data_dir: str, split: str = 'train', 
                 num_frames: int = 10, use_cache: bool = True):
        """
        Initialize AVQA dataset.
        
        Args:
            data_dir: Root directory of the dataset
            split: 'train', 'val', or 'test'
            num_frames: Number of frames to sample from each video
            use_cache: Whether to cache processed features
        """
        self.data_dir = data_dir
        self.split = split
        self.num_frames = num_frames
        self.use_cache = use_cache
        
        # Initialize parsers
        self.scene_parser = SceneGraphParser(use_blip=False)  # Use captions if provided
        self.query_parser = QueryGraphParser()
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Answer vocabulary
        self.answer_to_idx = self._build_answer_vocab()
        self.idx_to_answer = {v: k for k, v in self.answer_to_idx.items()}
        
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations."""
        anno_path = os.path.join(self.data_dir, f'{self.split}_annotations.json')
        
        if not os.path.exists(anno_path):
            # Create dummy annotations for demonstration
            print(f"Warning: Annotation file not found at {anno_path}")
            print("Creating dummy data for demonstration...")
            return self._create_dummy_data()
        
        with open(anno_path, 'r') as f:
            annotations = json.load(f)
        
        # Normalize field names (MUSIC-AVQA uses 'question_content')
        for anno in annotations:
            if 'question_content' in anno and 'question' not in anno:
                anno['question'] = anno['question_content']
        
        return annotations
    
    def _create_dummy_data(self, num_samples: int = 100) -> List[Dict]:
        """Create dummy data for demonstration purposes."""
        dummy_data = []
        question_types = [
            "What instrument is playing?",
            "How many instruments are visible?",
            "Where is the performer located?",
            "What is the tempo of the music?",
            "Is there a piano in the scene?"
        ]
        answers = ["piano", "guitar", "violin", "drums", "yes", "no", "two", "three", "left", "right"]
        
        for i in range(num_samples):
            sample = {
                'video_id': f'video_{i:04d}',
                'question': np.random.choice(question_types),
                'answer': np.random.choice(answers),
                'question_type': 'counting' if 'many' in np.random.choice(question_types) else 'existential'
            }
            dummy_data.append(sample)
        
        return dummy_data
    
    def _build_answer_vocab(self) -> Dict[str, int]:
        """Build vocabulary of answers."""
        answers = set()
        for anno in self.annotations:
            answers.add(anno['answer'])
        
        answer_to_idx = {ans: idx for idx, ans in enumerate(sorted(answers))}
        return answer_to_idx
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - audio_feat: [T, 128] audio features
            - visual_feat: [T, 512] visual features
            - question_feat: [1, 512] question features
            - sg_data: Scene graph data
            - qg_data: Query graph data
            - answer: Answer label
        """
        anno = self.annotations[idx]
        
        # Load or generate features
        audio_feat = self._load_audio_features(anno)
        visual_feat = self._load_visual_features(anno)
        question_feat = self._load_question_features(anno)
        
        # Generate scene graphs
        sg_data = self._generate_scene_graph(anno)
        qg_data = self._generate_query_graph(anno)
        
        # Get answer label
        answer = self.answer_to_idx[anno['answer']]
        
        return {
            'audio_feat': audio_feat,
            'visual_feat': visual_feat,
            'question_feat': question_feat,
            'sg_data': sg_data,
            'qg_data': qg_data,
            'answer': torch.tensor(answer, dtype=torch.long),
            'video_id': anno['video_id']
        }
    
    def _load_audio_features(self, anno: Dict) -> torch.Tensor:
        """Load audio features (VGGish: 128-dim)."""
        video_id = anno['video_id']
        feat_path = os.path.join(self.data_dir, 'audio_features', f'{video_id}.npy')
        
        if os.path.exists(feat_path):
            # Load preprocessed features
            features = np.load(feat_path)
            features = torch.from_numpy(features).float()
            # Ensure correct shape [num_frames, 128]
            if len(features.shape) == 1:
                features = features.unsqueeze(0).repeat(self.num_frames, 1)
            elif features.shape[0] != self.num_frames:
                # Interpolate to match num_frames
                features = torch.nn.functional.interpolate(
                    features.unsqueeze(0).transpose(1, 2),
                    size=self.num_frames,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2).squeeze(0)
            return features
        else:
            # Generate random features for demo
            return torch.randn(self.num_frames, 128)
    
    def _load_visual_features(self, anno: Dict) -> torch.Tensor:
        """Load visual features (CLIP: 512-dim)."""
        video_id = anno['video_id']
        feat_path = os.path.join(self.data_dir, 'visual_features', f'{video_id}.npy')
        
        if os.path.exists(feat_path):
            # Load preprocessed features
            features = np.load(feat_path)
            features = torch.from_numpy(features).float()
            # Ensure correct shape [num_frames, 512]
            if len(features.shape) == 1:
                features = features.unsqueeze(0).repeat(self.num_frames, 1)
            elif features.shape[0] != self.num_frames:
                # Interpolate to match num_frames
                features = torch.nn.functional.interpolate(
                    features.unsqueeze(0).transpose(1, 2),
                    size=self.num_frames,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2).squeeze(0)
            return features
        else:
            # Generate random features for demo
            return torch.randn(self.num_frames, 512)
    
    def _load_question_features(self, anno: Dict) -> torch.Tensor:
        """Load question features (Text encoder: 512-dim)."""
        # For now, use random features
        # TODO: Implement text encoder (BERT/CLIP) for questions
        return torch.randn(1, 512)
    
    def _generate_scene_graph(self, anno: Dict) -> Data:
        """Generate scene graph from video frames."""
        # For demo, create a simple graph
        # In real implementation, use captions or object detection
        caption = f"A person playing an instrument"
        graph = self.scene_parser.parse_caption(caption)
        data = self.scene_parser.graph_to_pyg_data(graph)
        return data
    
    def _generate_query_graph(self, anno: Dict) -> Data:
        """Generate query graph from question."""
        question = str(anno['question'])  # Convert to str to handle numpy.str_
        graph = self.query_parser.parse_query(question)
        data = self.query_parser.graph_to_pyg_data(graph)
        return data


def collate_avqa(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for AVQA dataset.
    Handles batching of graph data.
    """
    audio_feats = torch.stack([item['audio_feat'] for item in batch])
    visual_feats = torch.stack([item['visual_feat'] for item in batch])
    question_feats = torch.stack([item['question_feat'] for item in batch])
    answers = torch.stack([item['answer'] for item in batch])
    
    # Batch graph data
    sg_batch = Batch.from_data_list([item['sg_data'] for item in batch])
    qg_batch = Batch.from_data_list([item['qg_data'] for item in batch])
    
    return {
        'audio_feat': audio_feats,
        'visual_feat': visual_feats,
        'question_feat': question_feats,
        'sg_data': (sg_batch.x, sg_batch.edge_index, sg_batch.edge_attr, sg_batch.batch),
        'qg_data': (qg_batch.x, qg_batch.edge_index, qg_batch.edge_attr, qg_batch.batch),
        'answer': answers,
        'video_ids': [item['video_id'] for item in batch]
    }


def create_dataloaders(data_dir: str, batch_size: int = 4, num_workers: int = 4):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = AVQA_Dataset(data_dir, split='train')
    val_dataset = AVQA_Dataset(data_dir, split='val')
    test_dataset = AVQA_Dataset(data_dir, split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_avqa
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_avqa
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_avqa
    )
    
    return train_loader, val_loader, test_loader, train_dataset.idx_to_answer
