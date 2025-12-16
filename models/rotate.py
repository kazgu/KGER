"""
Fixed RotatE model implementation based on reference KnowledgeGraphEmbedding
Interface matches TransE for compatibility with the correction system
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional
import math


class RotatE(nn.Module):
    """RotatE model for knowledge graph embedding - fixed implementation"""
    
    def __init__(self, num_entities: int, num_relations: int, 
                 embedding_dim: int = 128, margin: float = 9.0, 
                 norm: int = 2, device: str = 'cpu', epsilon: float = 2.0):
        """
        Initialize RotatE model
        
        Args:
            num_entities: Number of entities in KG
            num_relations: Number of relations in KG
            embedding_dim: Dimension of embeddings (for each complex component)
            margin: Margin for ranking loss (gamma in reference)
            norm: Norm to use (kept for compatibility)
            device: Device to use
            epsilon: For initialization range calculation
        """
        super(RotatE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim  # This is the dimension per complex component
        self.margin = margin
        self.epsilon = epsilon
        self.device = torch.device(device)
        
        # Gamma parameter (equivalent to margin)
        self.gamma = nn.Parameter(
            torch.Tensor([margin]), 
            requires_grad=False
        )
        
        # Embedding range calculation (from reference implementation)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / embedding_dim]), 
            requires_grad=False
        )
        
        # Entity embeddings (complex numbers: 2 * embedding_dim)
        # RotatE requires double entity embedding
        self.entity_embeddings = nn.Parameter(torch.zeros(num_entities, embedding_dim * 2))
        nn.init.uniform_(
            self.entity_embeddings, 
            -self.embedding_range.item(), 
            self.embedding_range.item()
        )
        
        # Relation embeddings (phases - single dimension)
        self.relation_embeddings = nn.Parameter(torch.zeros(num_relations, embedding_dim))
        nn.init.uniform_(
            self.relation_embeddings, 
            -self.embedding_range.item(), 
            self.embedding_range.item()
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
            Positive scores (and negative scores if provided)
        """
        # Get embeddings
        head_emb = self.entity_embeddings[heads]  # [batch_size, 2*embedding_dim]
        relation_emb = self.relation_embeddings[relations]  # [batch_size, embedding_dim]
        tail_emb = self.entity_embeddings[tails]  # [batch_size, 2*embedding_dim]
        
        # Compute positive scores
        positive_score = self._rotate_score(head_emb.unsqueeze(1), 
                                          relation_emb.unsqueeze(1), 
                                          tail_emb.unsqueeze(1))
        positive_score = positive_score.squeeze(1)
        
        if negative_tails is None:
            return positive_score
        
        # Compute negative scores
        neg_tail_emb = self.entity_embeddings[negative_tails]
        negative_score = self._rotate_score(head_emb.unsqueeze(1),
                                          relation_emb.unsqueeze(1), 
                                          neg_tail_emb.unsqueeze(1))
        negative_score = negative_score.squeeze(1)
        
        return positive_score, negative_score
    
    def _rotate_score(self, head, relation, tail):
        """
        Compute RotatE scores using reference implementation
        
        Args:
            head: Head embeddings [batch_size, 1, 2*embedding_dim]
            relation: Relation embeddings [batch_size, 1, embedding_dim] 
            tail: Tail embeddings [batch_size, 1, 2*embedding_dim]
            
        Returns:
            Scores [batch_size, 1]
        """
        pi = 3.14159265358979323846
        
        # Split entity embeddings into real and imaginary parts
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        
        # Convert relation to phase (from reference implementation)
        phase_relation = relation / (self.embedding_range.item() / pi)
        
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        
        # Complex multiplication: head * relation
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        
        # Subtract tail: (head * relation) - tail
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        
        # Compute distance using reference implementation
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)  # L2 norm across real/imaginary
        score = self.gamma.item() - score.sum(dim=2)  # gamma - distance_sum
        
        return score
    
    def score_triple(self, head_idx: int, relation_idx: int, tail_idx: int) -> float:
        """Score a single triple"""
        with torch.no_grad():
            heads = torch.tensor([head_idx], device=self.device)
            relations = torch.tensor([relation_idx], device=self.device)
            tails = torch.tensor([tail_idx], device=self.device)
            
            score = self.forward(heads, relations, tails)
            # Convert to distance (lower is better) for compatibility
            return (self.gamma.item() - score.item())
    
    def predict_tail(self, head_idx: int, relation_idx: int, k: int = 10) -> Tuple[List[int], List[float]]:
        """Predict most likely tail entities"""
        with torch.no_grad():
            # Create batch with all entities as tails
            heads = torch.full((self.num_entities,), head_idx, device=self.device)
            relations = torch.full((self.num_entities,), relation_idx, device=self.device)
            tails = torch.arange(self.num_entities, device=self.device)
            
            scores = self.forward(heads, relations, tails)
            
            # Get top k (higher scores are better in this implementation)
            top_scores, top_indices = torch.topk(scores, k, largest=True)
            
        return top_indices.cpu().tolist(), top_scores.cpu().tolist()
    
    def predict_head(self, relation_idx: int, tail_idx: int, k: int = 10) -> Tuple[List[int], List[float]]:
        """Predict most likely head entities"""
        with torch.no_grad():
            heads = torch.arange(self.num_entities, device=self.device)
            relations = torch.full((self.num_entities,), relation_idx, device=self.device)
            tails = torch.full((self.num_entities,), tail_idx, device=self.device)
            
            scores = self.forward(heads, relations, tails)
            top_scores, top_indices = torch.topk(scores, k, largest=True)
            
        return top_indices.cpu().tolist(), top_scores.cpu().tolist()
    
    def predict_relation(self, head_idx: int, tail_idx: int, k: int = 10) -> Tuple[List[int], List[float]]:
        """Predict most likely relations"""
        with torch.no_grad():
            heads = torch.full((self.num_relations,), head_idx, device=self.device)
            relations = torch.arange(self.num_relations, device=self.device)
            tails = torch.full((self.num_relations,), tail_idx, device=self.device)
            
            scores = self.forward(heads, relations, tails)
            top_scores, top_indices = torch.topk(scores, k, largest=True)
            
        return top_indices.cpu().tolist(), top_scores.cpu().tolist()
    
    def loss_function(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using log-sigmoid (matching reference implementation)
        
        Args:
            positive_score: Scores for positive triples (higher is better)
            negative_score: Scores for negative triples (lower is better)
            
        Returns:
            Loss value
        """
        # Use log-sigmoid loss as in reference
        positive_loss = -F.logsigmoid(positive_score).mean()
        negative_loss = -F.logsigmoid(-negative_score).mean()
        
        return (positive_loss + negative_loss) / 2
    
    def get_embeddings(self, head_idx: int, relation_idx: int, tail_idx: int, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get embeddings for a triple (for compatibility with existing code)"""
        # 简化：直接返回embeddings，让调用者决定是否需要梯度
        # 这样避免了复杂的上下文管理和梯度冲突
        h_emb = self.entity_embeddings[head_idx]
        r_emb = self.relation_embeddings[relation_idx]
        t_emb = self.entity_embeddings[tail_idx]

        # For compatibility, expand relation to match entity dimension
        r_expanded = r_emb.repeat(2)  # Double the dimension

        return h_emb, r_expanded, t_emb
    
    def save(self, path: str):
        """Save model parameters"""
        torch.save({
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'embedding_dim': self.embedding_dim,
            'margin': self.margin,
            'epsilon': self.epsilon
        }, path)
        
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both tensor and state_dict formats
        if isinstance(checkpoint['entity_embeddings'], torch.Tensor):
            # Direct tensor format
            self.entity_embeddings.data = checkpoint['entity_embeddings'].to(self.device)
            self.relation_embeddings.data = checkpoint['relation_embeddings'].to(self.device)
        else:
            # State dict format (from nn.Embedding)
            entity_state = checkpoint['entity_embeddings']
            relation_state = checkpoint['relation_embeddings']
            
            # If it's a state_dict with 'weight' key
            if isinstance(entity_state, dict) and 'weight' in entity_state:
                self.entity_embeddings.data = entity_state['weight'].to(self.device)
            else:
                # Assume it's already the weight tensor
                self.entity_embeddings.data = entity_state.to(self.device)
                
            if isinstance(relation_state, dict) and 'weight' in relation_state:
                self.relation_embeddings.data = relation_state['weight'].to(self.device)
            else:
                self.relation_embeddings.data = relation_state.to(self.device)
        
        # Update parameters if they exist
        if 'embedding_dim' in checkpoint:
            self.embedding_dim = checkpoint['embedding_dim']
        if 'margin' in checkpoint:
            self.margin = checkpoint['margin']
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']