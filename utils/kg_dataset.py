"""
KG Dataset using torch.utils.data (matching reference implementation)
"""
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


class KGDataset(Dataset):
    """Knowledge Graph Dataset (from reference implementation)"""
    
    def __init__(self, data_path, entity2id, relation2id, noise_rate=0.0, noise_type='mixed', seed=None,
                 transe_model=None, score_based_noise=False):
        """
        Initialize dataset with optional noise injection
        
        Args:
            data_path: Path to JSON file with triples
            entity2id: Entity to ID mapping
            relation2id: Relation to ID mapping
            noise_rate: Percentage of triples to corrupt (0.0 to 1.0)
            noise_type: Type of corruption ('relation', 'entity', 'mixed')
            seed: Random seed for reproducibility
            transe_model: TransE model for score-based noise injection
            score_based_noise: If True, inject noise based on TransE scores
        """
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_entities = len(entity2id)
        self.num_relations = len(relation2id)
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        self.transe_model = transe_model
        self.score_based_noise = score_based_noise
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.triples = []
        self.original_triples = []  # Store original triples before corruption
        self.corrupted_indices = []  # Track which triples were corrupted
        with open(data_path, 'r') as f:
            data = json.load(f)
            for line in data:
                # Handle different formats
                if isinstance(line, dict) and 'triplet' in line:
                    h, r, t = line['triplet']
                else:
                    h, r, t = line
                
                # Convert to IDs
                if h in entity2id and r in relation2id and t in entity2id:
                    triple = (entity2id[h], relation2id[r], entity2id[t])
                    self.triples.append(triple)
                    self.original_triples.append(triple)
        
        # Apply noise if requested
        if self.noise_rate > 0:
            if self.score_based_noise and self.transe_model is not None:
                self._inject_score_based_noise()
            else:
                self._inject_noise()
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]
    
    def _inject_score_based_noise(self):
        """Inject noise based on TransE scores - corrupt worst scoring triples"""
        print(f"Injecting score-based noise to {self.noise_rate*100:.1f}% of data...")
        
        # Calculate scores for all triples
        triple_scores = []
        with torch.no_grad():
            for idx, (h, r, t) in enumerate(self.triples):
                score = self.transe_model.score_triple(h, r, t)
                triple_scores.append((idx, score))
        
        # Sort by score (descending - higher score means worse quality)
        triple_scores.sort(key=lambda x: x[1], reverse=False)
        
        # Select worst scoring triples to corrupt
        num_to_corrupt = int(len(self.triples) * self.noise_rate)
        self.corrupted_indices = [idx for idx, _ in triple_scores[:num_to_corrupt]]
        
        print(f"Selected {num_to_corrupt} worst-scoring triples (scores: {triple_scores[0][1]:.2f} to {triple_scores[num_to_corrupt-1][1]:.2f})")
        
        # Corrupt selected triples
        for idx in self.corrupted_indices:
            h, r, t = self.triples[idx]
            
            if self.noise_type == 'relation':
                # Corrupt relation
                new_r = random.randint(0, self.num_relations - 1)
                while new_r == r:
                    new_r = random.randint(0, self.num_relations - 1)
                self.triples[idx] = (h, new_r, t)
                
            elif self.noise_type == 'entity':
                # Randomly corrupt head or tail
                if random.random() < 0.5:
                    # Corrupt head
                    new_h = random.randint(0, self.num_entities - 1)
                    while new_h == h:
                        new_h = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (new_h, r, t)
                else:
                    # Corrupt tail
                    new_t = random.randint(0, self.num_entities - 1)
                    while new_t == t:
                        new_t = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (h, r, new_t)
                    
            else:  # mixed
                # Randomly choose what to corrupt
                corruption_choice = random.choice(['head', 'relation', 'tail'])
                if corruption_choice == 'head':
                    new_h = random.randint(0, self.num_entities - 1)
                    while new_h == h:
                        new_h = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (new_h, r, t)
                elif corruption_choice == 'relation':
                    new_r = random.randint(0, self.num_relations - 1)
                    while new_r == r:
                        new_r = random.randint(0, self.num_relations - 1)
                    self.triples[idx] = (h, new_r, t)
                else:  # tail
                    new_t = random.randint(0, self.num_entities - 1)
                    while new_t == t:
                        new_t = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (h, r, new_t)
    
    def _inject_noise(self):
        """Inject noise to the dataset (random selection)"""
        num_to_corrupt = int(len(self.triples) * self.noise_rate)
        self.corrupted_indices = random.sample(range(len(self.triples)), num_to_corrupt)
        
        for idx in self.corrupted_indices:
            h, r, t = self.triples[idx]
            
            if self.noise_type == 'relation':
                # Corrupt relation
                new_r = random.randint(0, self.num_relations - 1)
                while new_r == r:
                    new_r = random.randint(0, self.num_relations - 1)
                self.triples[idx] = (h, new_r, t)
                
            elif self.noise_type == 'entity':
                # Randomly corrupt head or tail
                if random.random() < 0.5:
                    # Corrupt head
                    new_h = random.randint(0, self.num_entities - 1)
                    while new_h == h:
                        new_h = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (new_h, r, t)
                else:
                    # Corrupt tail
                    new_t = random.randint(0, self.num_entities - 1)
                    while new_t == t:
                        new_t = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (h, r, new_t)
                    
            else:  # mixed
                # Randomly choose what to corrupt
                corruption_choice = random.choice(['head', 'relation', 'tail'])
                if corruption_choice == 'head':
                    new_h = random.randint(0, self.num_entities - 1)
                    while new_h == h:
                        new_h = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (new_h, r, t)
                elif corruption_choice == 'relation':
                    new_r = random.randint(0, self.num_relations - 1)
                    while new_r == r:
                        new_r = random.randint(0, self.num_relations - 1)
                    self.triples[idx] = (h, new_r, t)
                else:  # tail
                    new_t = random.randint(0, self.num_entities - 1)
                    while new_t == t:
                        new_t = random.randint(0, self.num_entities - 1)
                    self.triples[idx] = (h, r, new_t)
    
    def get_corruption_stats(self):
        """Get statistics about corruption"""
        return {
            'total_triples': len(self.triples),
            'corrupted_triples': len(self.corrupted_indices),
            'corruption_rate': len(self.corrupted_indices) / len(self.triples) if self.triples else 0,
            'corrupted_indices': self.corrupted_indices
        }


def load_id_mapping(file_path):
    """Load ID mapping from JSON file"""
    with open(file_path, 'r') as f:
        mapping = json.load(f)
    return mapping