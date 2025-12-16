"""
Advanced negative sampling strategies for TransE
"""
import torch
import numpy as np
from typing import Tuple, Set, Dict, List


class NegativeSampler:
    """Advanced negative sampling with various strategies"""
    
    def __init__(self, num_entities: int, num_relations: int, 
                 all_triples: Set[Tuple] = None):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.all_triples = all_triples or set()
        
        # Build entity type constraints for type-constrained sampling
        self.head_entities_per_relation = {}
        self.tail_entities_per_relation = {}
        if all_triples:
            for h, r, t in all_triples:
                if r not in self.head_entities_per_relation:
                    self.head_entities_per_relation[r] = set()
                    self.tail_entities_per_relation[r] = set()
                self.head_entities_per_relation[r].add(h)
                self.tail_entities_per_relation[r].add(t)
    
    def uniform_negative_sampling(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                  num_negatives: int = 1, 
                                  corrupt_rate: float = 0.5) -> List[Tuple]:
        """
        Uniform negative sampling (baseline)
        
        Args:
            batch: (heads, relations, tails) tensors
            num_negatives: Number of negatives per positive
            corrupt_rate: Probability of corrupting tail (vs head)
            
        Returns:
            List of negative samples
        """
        h, r, t = batch
        batch_size = h.shape[0]
        negatives = []
        
        for _ in range(num_negatives):
            neg_h = h.clone()
            neg_r = r.clone()
            neg_t = t.clone()
            
            for i in range(batch_size):
                if np.random.random() < corrupt_rate:
                    # Corrupt tail
                    new_t = np.random.randint(self.num_entities)
                    while new_t == t[i]:
                        new_t = np.random.randint(self.num_entities)
                    neg_t[i] = new_t
                else:
                    # Corrupt head
                    new_h = np.random.randint(self.num_entities)
                    while new_h == h[i]:
                        new_h = np.random.randint(self.num_entities)
                    neg_h[i] = new_h
            
            negatives.append((neg_h, neg_r, neg_t))
        
        return negatives
    
    def type_constrained_sampling(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                 num_negatives: int = 1) -> List[Tuple]:
        """
        Type-constrained negative sampling
        Sample negatives from entities that appear in the same position for the relation
        
        This is more challenging and helps the model learn better discrimination
        """
        h, r, t = batch
        batch_size = h.shape[0]
        negatives = []
        
        for _ in range(num_negatives):
            neg_h = h.clone()
            neg_r = r.clone() 
            neg_t = t.clone()
            
            for i in range(batch_size):
                rel_idx = r[i].item()
                
                if np.random.random() < 0.5:
                    # Corrupt tail with type-appropriate entity
                    if rel_idx in self.tail_entities_per_relation:
                        candidates = list(self.tail_entities_per_relation[rel_idx])
                        if len(candidates) > 1:
                            new_t = np.random.choice(candidates)
                            while new_t == t[i]:
                                new_t = np.random.choice(candidates)
                            neg_t[i] = new_t
                        else:
                            # Fallback to uniform
                            new_t = np.random.randint(self.num_entities)
                            while new_t == t[i]:
                                new_t = np.random.randint(self.num_entities)
                            neg_t[i] = new_t
                    else:
                        # Uniform sampling if no type info
                        new_t = np.random.randint(self.num_entities)
                        while new_t == t[i]:
                            new_t = np.random.randint(self.num_entities)
                        neg_t[i] = new_t
                else:
                    # Corrupt head with type-appropriate entity
                    if rel_idx in self.head_entities_per_relation:
                        candidates = list(self.head_entities_per_relation[rel_idx])
                        if len(candidates) > 1:
                            new_h = np.random.choice(candidates)
                            while new_h == h[i]:
                                new_h = np.random.choice(candidates)
                            neg_h[i] = new_h
                        else:
                            # Fallback to uniform
                            new_h = np.random.randint(self.num_entities)
                            while new_h == h[i]:
                                new_h = np.random.randint(self.num_entities)
                            neg_h[i] = new_h
                    else:
                        # Uniform sampling if no type info
                        new_h = np.random.randint(self.num_entities)
                        while new_h == h[i]:
                            new_h = np.random.randint(self.num_entities)
                        neg_h[i] = new_h
            
            negatives.append((neg_h, neg_r, neg_t))
        
        return negatives
    
    def self_adversarial_sampling(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  model, num_negatives: int = 1, 
                                  temperature: float = 1.0) -> Tuple:
        """
        Self-adversarial negative sampling
        Sample harder negatives based on current model scores
        
        Args:
            batch: (heads, relations, tails) tensors
            model: Current TransE model
            num_negatives: Number of negatives per positive
            temperature: Temperature for sampling distribution
            
        Returns:
            Negative samples and their weights
        """
        h, r, t = batch
        batch_size = h.shape[0]
        device = h.device
        
        # Generate candidate negatives
        neg_candidates = []
        for i in range(batch_size):
            # Generate multiple negative candidates
            candidates = []
            for _ in range(num_negatives * 10):  # Over-sample then select
                if np.random.random() < 0.5:
                    # Corrupt tail
                    new_t = np.random.randint(self.num_entities)
                    while new_t == t[i]:
                        new_t = np.random.randint(self.num_entities)
                    candidates.append((h[i].item(), r[i].item(), new_t))
                else:
                    # Corrupt head
                    new_h = np.random.randint(self.num_entities)
                    while new_h == h[i]:
                        new_h = np.random.randint(self.num_entities)
                    candidates.append((new_h, r[i].item(), t[i].item()))
            neg_candidates.append(candidates)
        
        # Score candidates with current model
        selected_negatives = []
        weights = []
        
        with torch.no_grad():
            for i in range(batch_size):
                cand_h = torch.tensor([c[0] for c in neg_candidates[i]], device=device)
                cand_r = torch.tensor([c[1] for c in neg_candidates[i]], device=device)
                cand_t = torch.tensor([c[2] for c in neg_candidates[i]], device=device)
                
                # Get scores from model
                scores = model(cand_h, cand_r, cand_t)
                
                # Convert to probabilities (lower score = higher probability)
                probs = torch.softmax(-scores / temperature, dim=0)
                
                # Sample based on probabilities
                indices = torch.multinomial(probs, num_negatives, replacement=False)
                
                for idx in indices:
                    selected_negatives.append(neg_candidates[i][idx])
                    weights.append(probs[idx].item())
        
        # Convert to tensors
        neg_h = torch.tensor([n[0] for n in selected_negatives], device=device)
        neg_r = torch.tensor([n[1] for n in selected_negatives], device=device)
        neg_t = torch.tensor([n[2] for n in selected_negatives], device=device)
        weights = torch.tensor(weights, device=device)
        
        return (neg_h, neg_r, neg_t), weights