"""
TransE model implementation for knowledge graph embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class TransE(nn.Module):
    """TransE model for knowledge graph embedding"""
    
    def __init__(self, num_entities: int, num_relations: int, 
                 embedding_dim: int = 128, margin: float = 1.0, 
                 norm: int = 1, device: str = 'cpu'):
        """
        Initialize TransE model
        
        Args:
            num_entities: Number of entities in KG
            num_relations: Number of relations in KG
            embedding_dim: Dimension of embeddings
            margin: Margin for ranking loss
            norm: Norm to use (1 or 2, default 1 for L1 norm)
            device: Device to use
        """
        super(TransE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin  # For hinge loss only
        self.norm = norm
        self.device = torch.device(device)
        self.epsilon = 2.0  # For embedding range calculation
        
        # Calculate embedding range (use a separate gamma value for initialization)
        gamma_init = 9.0  # Reference implementation's gamma value for initialization
        self.embedding_range = (gamma_init + self.epsilon) / embedding_dim
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Initialize with uniform distribution within embedding range
        nn.init.uniform_(
            self.entity_embeddings.weight.data,
            a=-self.embedding_range,
            b=self.embedding_range
        )
        nn.init.uniform_(
            self.relation_embeddings.weight.data,
            a=-self.embedding_range,
            b=self.embedding_range
        )
        
        # Normalize ONLY entity embeddings (as in reference)
        self.entity_embeddings.weight.data = F.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
        
        self.to(self.device)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor, 
                tails: torch.Tensor, negative_tails: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass - compute scores for triples
        
        Args:
            heads: Head entity indices
            relations: Relation indices
            tails: Tail entity indices
            negative_tails: Negative tail indices (optional)
            
        Returns:
            Positive distances (and negative distances if provided)
            Note: Lower distance = better match
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # Compute distance score (lower is better)
        score = h + r - t
        positive_distance = torch.norm(score, p=self.norm, dim=1)
        
        if negative_tails is None:
            return positive_distance
        
        # Compute negative distance if provided
        neg_t = self.entity_embeddings(negative_tails)
        neg_score = h + r - neg_t
        negative_distance = torch.norm(neg_score, p=self.norm, dim=1)
        
        return positive_distance, negative_distance
    
    def normalize_embeddings(self):
        """Normalize ONLY entity embeddings (as in reference)"""
        with torch.no_grad():
            # Normalize entity embeddings to unit sphere
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
            # DO NOT normalize relation embeddings
    
    def score_triple(self, head_idx: int, relation_idx: int, tail_idx: int) -> float:
        """
        Score a single triple
        
        Args:
            head_idx: Head entity index
            relation_idx: Relation index
            tail_idx: Tail entity index
            
        Returns:
            Distance score (lower is better)
        """
        with torch.no_grad():
            h = self.entity_embeddings(torch.tensor([head_idx], device=self.device))
            r = self.relation_embeddings(torch.tensor([relation_idx], device=self.device))
            t = self.entity_embeddings(torch.tensor([tail_idx], device=self.device))
            
            score = torch.norm(h + r - t, p=self.norm, dim=1)
            
        return score.item()
    
    def predict_tail(self, head_idx: int, relation_idx: int, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Predict most likely tail entities given head and relation
        
        Args:
            head_idx: Head entity index
            relation_idx: Relation index
            k: Number of candidates to return
            
        Returns:
            Top k tail entity indices and their scores
        """
        with torch.no_grad():
            h = self.entity_embeddings(torch.tensor([head_idx], device=self.device))
            r = self.relation_embeddings(torch.tensor([relation_idx], device=self.device))
            
            # Compute scores for all entities as tail
            all_entities = self.entity_embeddings.weight
            
            scores = torch.norm(h + r - all_entities.unsqueeze(0), p=self.norm, dim=2).squeeze()
            
            # Get top k entities with lowest scores (best matches)
            top_scores, top_indices = torch.topk(scores, k, largest=False)
            
        return top_indices.cpu().tolist(), top_scores.cpu().tolist()
    
    def predict_head(self, relation_idx: int, tail_idx: int, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Predict most likely head entities given relation and tail
        
        Args:
            relation_idx: Relation index
            tail_idx: Tail entity index
            k: Number of candidates to return
            
        Returns:
            Top k head entity indices and their scores
        """
        with torch.no_grad():
            r = self.relation_embeddings(torch.tensor([relation_idx], device=self.device))
            t = self.entity_embeddings(torch.tensor([tail_idx], device=self.device))
            
            # Compute scores for all entities as head
            all_entities = self.entity_embeddings.weight
            
            scores = torch.norm(all_entities.unsqueeze(0) + r - t, p=self.norm, dim=2).squeeze()
            
            # Get top k entities with lowest scores (best matches)
            top_scores, top_indices = torch.topk(scores, k, largest=False)
            
        return top_indices.cpu().tolist(), top_scores.cpu().tolist()
    
    def predict_relation(self, head_idx: int, tail_idx: int, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Predict most likely relations given head and tail
        
        Args:
            head_idx: Head entity index
            tail_idx: Tail entity index
            k: Number of candidates to return
            
        Returns:
            Top k relation indices and their scores
        """
        with torch.no_grad():
            h = self.entity_embeddings(torch.tensor([head_idx], device=self.device))
            t = self.entity_embeddings(torch.tensor([tail_idx], device=self.device))
            
            # Compute scores for all relations
            all_relations = self.relation_embeddings.weight
            
            scores = torch.norm(h.unsqueeze(0) + all_relations.unsqueeze(0) - t, p=self.norm, dim=2).squeeze()
            
            # Get top k relations with lowest scores (best matches)
            top_scores, top_indices = torch.topk(scores, k, largest=False)
            
        return top_indices.cpu().tolist(), top_scores.cpu().tolist()
    
    def loss_function(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """
        Compute hinge loss
        
        Args:
            positive_score: Distance scores for positive triples (lower is better)
            negative_score: Distance scores for negative triples (lower is better)
            
        Returns:
            Loss value
        """
        # Standard hinge loss: we want positive_score < negative_score
        # Loss = max(0, margin + positive_score - negative_score)
        return torch.mean(torch.max(torch.zeros_like(positive_score),
                                    self.margin + positive_score - negative_score))
    
    def get_embeddings(self, head_idx: int, relation_idx: int, tail_idx: int, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get embeddings for a triple
        
        Args:
            head_idx: Head entity index
            relation_idx: Relation index
            tail_idx: Tail entity index
            training: Whether in training mode (for compatibility with RotatE)
            
        Returns:
            Head, relation, and tail embeddings
        """
        # 简化：直接返回embeddings，让调用者决定是否需要梯度
        # 这样避免了梯度冲突，adapter可以正常工作
        h = self.entity_embeddings(torch.tensor([head_idx], device=self.device))
        r = self.relation_embeddings(torch.tensor([relation_idx], device=self.device))
        t = self.entity_embeddings(torch.tensor([tail_idx], device=self.device))
            
        return h.squeeze(0), r.squeeze(0), t.squeeze(0)
    
    def save(self, path: str):
        """Save model parameters"""
        torch.save({
            'entity_embeddings': self.entity_embeddings.state_dict(),
            'relation_embeddings': self.relation_embeddings.state_dict(),
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'embedding_dim': self.embedding_dim,
            'margin': self.margin,
            'norm': self.norm
        }, path)
        
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        self.entity_embeddings.load_state_dict(checkpoint['entity_embeddings'])
        self.relation_embeddings.load_state_dict(checkpoint['relation_embeddings'])