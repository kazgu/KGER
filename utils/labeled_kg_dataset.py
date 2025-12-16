"""
Labeled KG Dataset with error type annotations
"""
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


class LabeledKGDataset(Dataset):
    """Knowledge Graph Dataset with labels for error detection"""
    
    def __init__(self, data_path, entity2id, relation2id, 
                 noise_rate=0.0, noise_type='mixed', seed=None):
        """
        Initialize labeled dataset with noise injection and error tracking
        
        Args:
            data_path: Path to JSON file with triples
            entity2id: Entity to ID mapping
            relation2id: Relation to ID mapping
            noise_rate: Percentage of triples to corrupt (0.0 to 1.0)
            noise_type: Type of corruption ('relation', 'entity', 'mixed')
            seed: Random seed for reproducibility
        """
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_entities = len(entity2id)
        self.num_relations = len(relation2id)
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Data storage
        self.triples = []  # Current triples (may be corrupted)
        self.original_triples = []  # Original correct triples
        self.labels = []  # 0 = correct, 1 = corrupted
        self.error_types = []  # 'none', 'head', 'relation', 'tail'
        self.corrupted_indices = []  # Indices of corrupted triples
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
            for line in data:
                if isinstance(line, dict) and 'triplet' in line:
                    h, r, t = line['triplet']
                else:
                    h, r, t = line
                
                if h in entity2id and r in relation2id and t in entity2id:
                    triple = (entity2id[h], relation2id[r], entity2id[t])
                    self.triples.append(triple)
                    self.original_triples.append(triple)
                    self.labels.append(0)  # Initially all correct
                    self.error_types.append('none')
        
        # Inject noise with labels
        if self.noise_rate > 0:
            self._inject_labeled_noise()
    
    def _inject_labeled_noise(self):
        """Inject noise and track error types"""
        num_to_corrupt = int(len(self.triples) * self.noise_rate)
        self.corrupted_indices = random.sample(range(len(self.triples)), num_to_corrupt)
        
        print(f"Injecting labeled noise to {num_to_corrupt}/{len(self.triples)} triples...")
        
        for idx in tqdm(self.corrupted_indices):
            h, r, t = self.triples[idx]
            
            # Determine corruption type
            if self.noise_type == 'relation':
                corruption_choice = 'relation'
            elif self.noise_type == 'entity':
                corruption_choice = random.choice(['head', 'tail'])
            else:  # mixed
                corruption_choice = random.choice(['head', 'relation', 'tail'])
            
            # Apply corruption
            if corruption_choice == 'head':
                new_h = random.randint(0, self.num_entities - 1)
                while new_h == h:
                    new_h = random.randint(0, self.num_entities - 1)
                self.triples[idx] = (new_h, r, t)
                self.error_types[idx] = 'head'
                
            elif corruption_choice == 'relation':
                new_r = random.randint(0, self.num_relations - 1)
                while new_r == r:
                    new_r = random.randint(0, self.num_relations - 1)
                self.triples[idx] = (h, new_r, t)
                self.error_types[idx] = 'relation'
                
            else:  # tail
                new_t = random.randint(0, self.num_entities - 1)
                while new_t == t:
                    new_t = random.randint(0, self.num_entities - 1)
                self.triples[idx] = (h, r, new_t)
                self.error_types[idx] = 'tail'
            
            # Mark as corrupted
            self.labels[idx] = 1
        
        # Print statistics
        error_counts = {'head': 0, 'relation': 0, 'tail': 0}
        for error_type in self.error_types:
            if error_type != 'none':
                error_counts[error_type] += 1
        
        print(f"Error distribution: {error_counts}")
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Get item with label and error type
        
        Returns:
            tuple: (triple, label, error_type, original_triple)
        """
        return (
            self.triples[idx],
            self.labels[idx],
            self.error_types[idx],
            self.original_triples[idx]
        )
    
    def get_triple_with_label(self, idx):
        """Get triple with its label"""
        return self.triples[idx], self.labels[idx]
    
    def get_error_type(self, idx):
        """Get error type for a triple"""
        return self.error_types[idx]
    
    def get_original_triple(self, idx):
        """Get original correct triple"""
        return self.original_triples[idx]
    
    def get_corruption_stats(self):
        """Get statistics about corruption"""
        total_corrupted = sum(self.labels)
        error_type_counts = {}
        for error_type in self.error_types:
            if error_type not in error_type_counts:
                error_type_counts[error_type] = 0
            error_type_counts[error_type] += 1
        
        return {
            'total_triples': len(self.triples),
            'corrupted_triples': total_corrupted,
            'corruption_rate': total_corrupted / len(self.triples) if self.triples else 0,
            'error_type_distribution': error_type_counts,
            'corrupted_indices': self.corrupted_indices
        }
    
    def get_error_type_encoding(self, idx):
        """
        Get one-hot encoding of error type
        
        Returns:
            tensor: [is_correct, is_head_error, is_relation_error, is_tail_error]
        """
        error_type = self.error_types[idx]
        encoding = torch.zeros(4)
        
        if error_type == 'none':
            encoding[0] = 1.0  # is_correct
        elif error_type == 'head':
            encoding[1] = 1.0  # is_head_error
        elif error_type == 'relation':
            encoding[2] = 1.0  # is_relation_error
        elif error_type == 'tail':
            encoding[3] = 1.0  # is_tail_error
        
        return encoding