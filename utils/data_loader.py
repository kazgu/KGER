"""
Data loader for knowledge graph datasets
"""
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import random
from .kg_dataset import KGDataset  # Import from kg_dataset.py


class KGDataLoader:
    """Data loader for knowledge graph"""
    
    def __init__(self, data_path: str):
        """
        Initialize data loader
        
        Args:
            data_path: Path to dataset directory
        """
        self.data_path = data_path
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        self.triples = []
        self.train_triples = []
        self.valid_triples = []
        self.test_triples = []
        
        # Create sets for fast lookup
        self.triple_set = set()
        self.train_set = set()
        self.valid_set = set()
        self.test_set = set()
        
        self._load_mappings()
        self._load_triples()
        
    def _load_mappings(self):
        """Load entity and relation mappings"""
        # Load entity mappings
        with open(f"{self.data_path}/entity2id.json", 'r') as f:
            self.entity2id = json.load(f)
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        
        # Load relation mappings
        with open(f"{self.data_path}/relation2id.json", 'r') as f:
            self.relation2id = json.load(f)
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        
    def _load_triples(self):
        """Load triples from files"""
        # Try to load train/valid/test splits first
        train_loaded = False
        valid_loaded = False
        test_loaded = False
        
        # Load train triples
        try:
            with open(f"{self.data_path}/train.json", 'r') as f:
                train_data = json.load(f)
                
            for triple in train_data:
                if isinstance(triple, dict) and 'triplet' in triple:
                    h, r, t = triple['triplet']
                else:
                    h, r, t = triple
                    
                h_id = self.entity2id.get(h, -1)
                r_id = self.relation2id.get(r, -1)
                t_id = self.entity2id.get(t, -1)
                
                if h_id != -1 and r_id != -1 and t_id != -1:
                    self.train_triples.append((h_id, r_id, t_id))
                    self.train_set.add((h_id, r_id, t_id))
            train_loaded = True
        except FileNotFoundError:
            pass
            
        # Load valid triples
        try:
            with open(f"{self.data_path}/valid.json", 'r') as f:
                valid_data = json.load(f)
                
            for triple in valid_data:
                if isinstance(triple, dict) and 'triplet' in triple:
                    h, r, t = triple['triplet']
                else:
                    h, r, t = triple
                    
                h_id = self.entity2id.get(h, -1)
                r_id = self.relation2id.get(r, -1)
                t_id = self.entity2id.get(t, -1)
                
                if h_id != -1 and r_id != -1 and t_id != -1:
                    self.valid_triples.append((h_id, r_id, t_id))
                    self.valid_set.add((h_id, r_id, t_id))
            valid_loaded = True
        except FileNotFoundError:
            pass
            
        # Load test triples
        try:
            with open(f"{self.data_path}/test.json", 'r') as f:
                test_data = json.load(f)
                
            for triple in test_data:
                if isinstance(triple, dict) and 'triplet' in triple:
                    h, r, t = triple['triplet']
                else:
                    h, r, t = triple
                    
                h_id = self.entity2id.get(h, -1)
                r_id = self.relation2id.get(r, -1)
                t_id = self.entity2id.get(t, -1)
                
                if h_id != -1 and r_id != -1 and t_id != -1:
                    self.test_triples.append((h_id, r_id, t_id))
                    self.test_set.add((h_id, r_id, t_id))
            test_loaded = True
        except FileNotFoundError:
            pass
            
        # If we have train/valid/test, combine them for all triples
        if train_loaded or valid_loaded or test_loaded:
            self.triples = self.train_triples + self.valid_triples + self.test_triples
            self.triple_set = self.train_set | self.valid_set | self.test_set
        else:
            # Try to load from triplets.json as fallback
            try:
                with open(f"{self.data_path}/triplets.json", 'r') as f:
                    triplets_data = json.load(f)
                    
                for triple in triplets_data:
                    if isinstance(triple, dict) and 'triplet' in triple:
                        h, r, t = triple['triplet']
                    else:
                        h, r, t = triple
                        
                    h_id = self.entity2id.get(h, -1)
                    r_id = self.relation2id.get(r, -1)
                    t_id = self.entity2id.get(t, -1)
                    
                    if h_id != -1 and r_id != -1 and t_id != -1:
                        self.triples.append((h_id, r_id, t_id))
                        self.triple_set.add((h_id, r_id, t_id))
                        
                # If no splits, use all triples for training
                if not train_loaded:
                    self.train_triples = self.triples
                    self.train_set = self.triple_set.copy()
            except FileNotFoundError:
                print("Warning: No triple files found!")
            
    def get_num_entities(self) -> int:
        """Get number of entities"""
        return len(self.entity2id)
    
    def get_num_relations(self) -> int:
        """Get number of relations"""
        return len(self.relation2id)
    
    def get_train_dataloader(self, batch_size: int, shuffle: bool = True, num_workers: int = 4, 
                             noise_rate: float = 0.0, noise_type: str = 'mixed', seed: int = None,
                             transe_model=None, score_based_noise: bool = False) -> DataLoader:
        """Get training dataloader with optional noise injection
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
            noise_rate: Percentage of triples to corrupt (0.0 to 1.0)
            noise_type: Type of corruption ('relation', 'entity', 'mixed')
            seed: Random seed for reproducibility
            transe_model: TransE model for score-based noise injection
            score_based_noise: If True, inject noise based on TransE scores
        """
        dataset = KGDataset(f"{self.data_path}/train.json", self.entity2id, self.relation2id,
                          noise_rate=noise_rate, noise_type=noise_type, seed=seed,
                          transe_model=transe_model, score_based_noise=score_based_noise)
        if noise_rate > 0:
            stats = dataset.get_corruption_stats()
            print(f"  Injected noise to training data: {stats['corrupted_triples']}/{stats['total_triples']} ({stats['corruption_rate']:.1%})")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def get_test_dataloader(self, batch_size: int, shuffle: bool = False, num_workers: int = 4,
                            noise_rate: float = 0.0, noise_type: str = 'mixed', seed: int = None,
                            transe_model=None, score_based_noise: bool = False) -> DataLoader:
        """Get test dataloader with optional noise injection
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
            noise_rate: Percentage of triples to corrupt (0.0 to 1.0)
            noise_type: Type of corruption ('relation', 'entity', 'mixed')
            seed: Random seed for reproducibility
            transe_model: TransE model for score-based noise injection
            score_based_noise: If True, inject noise based on TransE scores
        """
        dataset = KGDataset(f"{self.data_path}/test.json", self.entity2id, self.relation2id,
                          noise_rate=noise_rate, noise_type=noise_type, seed=seed,
                          transe_model=transe_model, score_based_noise=score_based_noise)
        if noise_rate > 0:
            stats = dataset.get_corruption_stats()
            print(f"  Injected noise to test data: {stats['corrupted_triples']}/{stats['total_triples']} ({stats['corruption_rate']:.1%})")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def get_valid_dataloader(self, batch_size: int, shuffle: bool = False, num_workers: int = 4,
                            noise_rate: float = 0.0, noise_type: str = 'mixed', seed: int = None,
                            transe_model=None, score_based_noise: bool = False) -> DataLoader:
        """Get validation dataloader with optional noise injection"""
        dataset = KGDataset(f"{self.data_path}/valid.json", self.entity2id, self.relation2id,
                          noise_rate=noise_rate, noise_type=noise_type, seed=seed,
                          transe_model=transe_model, score_based_noise=score_based_noise)
        if noise_rate > 0:
            stats = dataset.get_corruption_stats()
            print(f"  Injected noise to valid data: {stats['corrupted_triples']}/{stats['total_triples']} ({stats['corruption_rate']:.1%})")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def get_all_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Get dataloader for all triples"""
        dataset = KGDataset(self.triples)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def sample_negative_triple(self, positive_triple: Tuple[int, int, int], 
                              mode: str = 'random') -> Tuple[int, int, int]:
        """
        Sample a negative triple
        
        Args:
            positive_triple: Positive triple (h, r, t)
            mode: Sampling mode ('random', 'corrupt_head', 'corrupt_tail')
            
        Returns:
            Negative triple
        """
        h, r, t = positive_triple
        num_entities = len(self.entity2id)
        
        if mode == 'corrupt_head':
            # Corrupt head entity
            while True:
                new_h = random.randint(0, num_entities - 1)
                if (new_h, r, t) not in self.triple_set:
                    return (new_h, r, t)
                    
        elif mode == 'corrupt_tail':
            # Corrupt tail entity
            while True:
                new_t = random.randint(0, num_entities - 1)
                if (h, r, new_t) not in self.triple_set:
                    return (h, r, new_t)
                    
        else:  # random
            # Randomly corrupt head or tail
            if random.random() < 0.5:
                return self.sample_negative_triple(positive_triple, 'corrupt_head')
            else:
                return self.sample_negative_triple(positive_triple, 'corrupt_tail')
    
    def get_batch_with_negatives(self, batch_size: int, 
                                 num_negatives: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of positive and negative triples
        
        Args:
            batch_size: Batch size
            num_negatives: Number of negative samples per positive
            
        Returns:
            Positive and negative triple tensors
            - positive: [batch_size, 3]
            - negative: [batch_size * num_negatives, 3]
        """
        # Sample positive triples
        # Allow replacement if batch_size > number of train triples
        replace = batch_size > len(self.train_triples)
        positive_indices = np.random.choice(len(self.train_triples), batch_size, replace=replace)
        positive_triples = [self.train_triples[i] for i in positive_indices]
        
        # Sample negative triples
        negative_triples = []
        for pos_triple in positive_triples:
            for _ in range(num_negatives):
                neg_triple = self.sample_negative_triple(pos_triple)
                negative_triples.append(neg_triple)
        
        # Convert to tensors
        pos_tensor = torch.tensor(positive_triples, dtype=torch.long)
        neg_tensor = torch.tensor(negative_triples, dtype=torch.long)
        
        return pos_tensor, neg_tensor
    
    def triple_to_names(self, triple: Tuple[int, int, int]) -> Tuple[str, str, str]:
        """
        Convert triple IDs to names
        
        Args:
            triple: Triple with IDs (h_id, r_id, t_id)
            
        Returns:
            Triple with names (h_name, r_name, t_name)
        """
        h_id, r_id, t_id = triple
        h_name = self.id2entity.get(h_id, f"Entity_{h_id}")
        r_name = self.id2relation.get(r_id, f"Relation_{r_id}")
        t_name = self.id2entity.get(t_id, f"Entity_{t_id}")
        
        return (h_name, r_name, t_name)
