"""
Scene graph parser for extracting structured representations from images and text.
Uses BLIP for image captioning and sng_parser for scene graph generation.
"""

import torch
import requests
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import sng_parser


class SceneGraphParser:
    """
    Parser to convert images or captions to scene graphs.
    """
    def __init__(self, use_blip=True, blip_model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize the scene graph parser.
        
        Args:
            use_blip: Whether to use BLIP for image captioning
            blip_model_name: Name of the BLIP model to use
        """
        self.use_blip = use_blip
        self.blip_model_name = blip_model_name
        
        if use_blip:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
                self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
                self.blip_model.eval()
            except ImportError:
                print("Warning: transformers not installed. Install with: pip install transformers")
                self.use_blip = False

    def caption_image(self, image):
        """
        Generate caption for an image using BLIP.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Generated caption string
        """
        if not self.use_blip:
            return "A scene with objects and relationships."
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        inputs = self.blip_processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.blip_model.generate(**inputs)
        
        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def parse_caption(self, caption: str) -> Dict:
        """
        Parse a caption into a scene graph.
        
        Args:
            caption: Text caption
            
        Returns:
            Scene graph dictionary with entities and relations
        """
        # Ensure caption is a regular Python string
        caption = str(caption)
        
        try:
            graph = sng_parser.parse(caption)
        except Exception as e:
            # Fallback to simple graph if parsing fails
            print(f"Warning: Failed to parse caption '{caption}': {e}")
            graph = {'entities': [{'head': 'object', 'span': [0, len(caption)]}], 'relations': []}
        
        return graph

    def parse_image(self, image) -> Dict:
        """
        Parse an image into a scene graph.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Scene graph dictionary
        """
        caption = self.caption_image(image)
        return self.parse_caption(caption)

    def graph_to_pyg_data(self, graph: Dict, node_embedding_fn=None, edge_embedding_fn=None):
        """
        Convert scene graph to PyTorch Geometric Data format.
        
        Args:
            graph: Scene graph dictionary from sng_parser
            node_embedding_fn: Function to embed node labels (optional)
            edge_embedding_fn: Function to embed edge labels (optional)
            
        Returns:
            Dictionary with x (node features), edge_index, edge_attr
        """
        from torch_geometric.data import Data
        
        # Extract entities (nodes)
        entities = graph.get('entities', [])
        
        # Create node features
        node_features = []
        node_labels = []
        
        for i, entity in enumerate(entities):
            label = entity.get('head', 'unknown')
            node_labels.append(label)
            
            if node_embedding_fn:
                feat = node_embedding_fn(label)
            else:
                # Default: random embedding
                feat = torch.randn(512)
            
            node_features.append(feat)
        
        if len(node_features) == 0:
            # Empty graph - create dummy node
            node_features = [torch.randn(512)]
            node_labels = ['unknown']
        
        x = torch.stack(node_features)
        
        # Extract relations (edges)
        relations = graph.get('relations', [])
        edge_index = []
        edge_attrs = []
        edge_labels = []
        
        for relation in relations:
            subject_idx = relation.get('subject', 0)
            object_idx = relation.get('object', 0)
            relation_label = relation.get('relation', 'related_to')
            
            # Ensure indices are valid
            if subject_idx < len(node_labels) and object_idx < len(node_labels):
                edge_index.append([subject_idx, object_idx])
                edge_labels.append(relation_label)
                
                if edge_embedding_fn:
                    edge_feat = edge_embedding_fn(relation_label)
                else:
                    edge_feat = torch.randn(512)
                
                edge_attrs.append(edge_feat)
        
        if len(edge_index) == 0:
            # No edges - create self-loop
            edge_index = [[0, 0]]
            edge_attrs = [torch.randn(512)]
            edge_labels = ['self']
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.stack(edge_attrs)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.node_labels = node_labels
        data.edge_labels = edge_labels
        
        return data


class QueryGraphParser:
    """
    Parser to convert questions into query graphs.
    """
    def __init__(self):
        pass

    def parse_query(self, query: str) -> Dict:
        """
        Parse a query text into a graph structure.
        
        Args:
            query: Question string
            
        Returns:
            Query graph dictionary
        """
        # Ensure query is a regular Python string
        query = str(query)
        
        try:
            graph = sng_parser.parse(query)
        except Exception as e:
            # Fallback to simple graph if parsing fails
            print(f"Warning: Failed to parse query '{query}': {e}")
            graph = {'entities': [{'head': 'query', 'span': [0, len(query)]}], 'relations': []}
        
        return graph

    def graph_to_pyg_data(self, graph: Dict, node_embedding_fn=None, edge_embedding_fn=None):
        """
        Convert query graph to PyTorch Geometric Data format.
        Similar to SceneGraphParser.graph_to_pyg_data
        """
        from torch_geometric.data import Data
        
        entities = graph.get('entities', [])
        
        node_features = []
        node_labels = []
        
        for entity in entities:
            label = entity.get('head', 'unknown')
            node_labels.append(label)
            
            if node_embedding_fn:
                feat = node_embedding_fn(label)
            else:
                feat = torch.randn(512)
            
            node_features.append(feat)
        
        if len(node_features) == 0:
            node_features = [torch.randn(512)]
            node_labels = ['query']
        
        x = torch.stack(node_features)
        
        relations = graph.get('relations', [])
        edge_index = []
        edge_attrs = []
        edge_labels = []
        
        for relation in relations:
            subject_idx = relation.get('subject', 0)
            object_idx = relation.get('object', 0)
            relation_label = relation.get('relation', 'related_to')
            
            if subject_idx < len(node_labels) and object_idx < len(node_labels):
                edge_index.append([subject_idx, object_idx])
                edge_labels.append(relation_label)
                
                if edge_embedding_fn:
                    edge_feat = edge_embedding_fn(relation_label)
                else:
                    edge_feat = torch.randn(512)
                
                edge_attrs.append(edge_feat)
        
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            edge_attrs = [torch.randn(512)]
            edge_labels = ['self']
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_attr = torch.stack(edge_attrs)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.node_labels = node_labels
        data.edge_labels = edge_labels
        
        return data
