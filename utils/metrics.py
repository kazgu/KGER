"""
Evaluation metrics for knowledge graph embeddings
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm


class KGEvaluator:
    """Evaluator for knowledge graph embedding models"""
    
    def __init__(self, model, data_loader, device):
        """
        Initialize evaluator
        
        Args:
            model: TransE model
            data_loader: KG data loader
            device: Device to use
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        
    def evaluate_link_prediction(self, test_triples: List[Tuple[int, int, int]], 
                                filtered: bool = True) -> Dict[str, float]:
        """
        Evaluate link prediction performance
        
        Args:
            test_triples: Test triples
            filtered: Whether to use filtered metrics
            
        Returns:
            Dictionary of metrics
        """
        ranks = []
        reciprocal_ranks = []
        hits_at_k = {1: 0, 3: 0, 10: 0}
        
        # Process in batches for efficiency
        batch_size = 100
        num_batches = (len(test_triples) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(test_triples))
            batch_triples = test_triples[batch_start:batch_end]
            
            for h, r, t in batch_triples:
                # Predict tail
                tail_rank = self._get_rank(h, r, t, 'tail', filtered)
                ranks.append(tail_rank)
                reciprocal_ranks.append(1.0 / tail_rank)
                
                for k in [1, 3, 10]:
                    if tail_rank <= k:
                        hits_at_k[k] += 1
                
                # Predict head
                head_rank = self._get_rank(h, r, t, 'head', filtered)
                ranks.append(head_rank)
                reciprocal_ranks.append(1.0 / head_rank)
                
                for k in [1, 3, 10]:
                    if head_rank <= k:
                        hits_at_k[k] += 1
        
        # Calculate metrics
        num_predictions = len(test_triples) * 2  # Both head and tail predictions
        
        metrics = {
            'mrr': np.mean(reciprocal_ranks),
            'mean_rank': np.mean(ranks),
            'hits@1': hits_at_k[1] / num_predictions,
            'hits@3': hits_at_k[3] / num_predictions,
            'hits@10': hits_at_k[10] / num_predictions
        }
        
        return metrics
    
    def _get_rank(self, h: int, r: int, t: int, mode: str, filtered: bool) -> int:
        """
        Get rank of correct entity
        
        Args:
            h: Head entity
            r: Relation
            t: Tail entity
            mode: 'head' or 'tail'
            filtered: Whether to filter out other correct triples
            
        Returns:
            Rank of correct entity
        """
        with torch.no_grad():
            if mode == 'tail':
                # Score all possible tails
                h_tensor = torch.tensor([h], device=self.device)
                r_tensor = torch.tensor([r], device=self.device)
                all_entities = torch.arange(self.model.num_entities, device=self.device)
                
                h_emb = self.model.entity_embeddings(h_tensor)
                r_emb = self.model.relation_embeddings(r_tensor)
                all_t_emb = self.model.entity_embeddings(all_entities)
                
                scores = torch.norm(h_emb + r_emb - all_t_emb, p=self.model.norm, dim=1)
                target_score = scores[t].item()
                
            else:  # mode == 'head'
                # Score all possible heads
                r_tensor = torch.tensor([r], device=self.device)
                t_tensor = torch.tensor([t], device=self.device)
                all_entities = torch.arange(self.model.num_entities, device=self.device)
                
                all_h_emb = self.model.entity_embeddings(all_entities)
                r_emb = self.model.relation_embeddings(r_tensor)
                t_emb = self.model.entity_embeddings(t_tensor)
                
                scores = torch.norm(all_h_emb + r_emb - t_emb, p=self.model.norm, dim=1)
                target_score = scores[h].item()
            
            # Filter out other correct triples if requested
            if filtered:
                if mode == 'tail':
                    # Get all correct tails for (h, r, ?)
                    correct_tails = self._get_filtered_entities(h, r, mode='tail')
                    # Set their scores to infinity except for the target
                    for correct_t in correct_tails:
                        if correct_t != t:
                            scores[correct_t] = float('inf')
                else:
                    # Get all correct heads for (?, r, t)
                    correct_heads = self._get_filtered_entities(r, t, mode='head')
                    # Set their scores to infinity except for the target
                    for correct_h in correct_heads:
                        if correct_h != h:
                            scores[correct_h] = float('inf')
            
            # Count how many entities have better or equal score
            rank = (scores <= target_score).sum().item()
            
        return rank
    
    def _get_filtered_entities(self, e1: int, e2: int, mode: str) -> List[int]:
        """
        Get filtered entities for a given query
        
        Args:
            e1: First entity (head if mode='tail', relation if mode='head')
            e2: Second entity (relation if mode='tail', tail if mode='head')
            mode: 'head' or 'tail'
            
        Returns:
            List of correct entities
        """
        correct_entities = []
        
        all_triples = (self.data_loader.train_triples + 
                      self.data_loader.valid_triples + 
                      self.data_loader.test_triples)
        
        for h, r, t in all_triples:
            if mode == 'tail' and h == e1 and r == e2:
                correct_entities.append(t)
            elif mode == 'head' and r == e1 and t == e2:
                correct_entities.append(h)
        
        return list(set(correct_entities))
    
    def evaluate_triple_quality(self, triples: List[Tuple[int, int, int]]) -> Dict[str, float]:
        """
        Evaluate quality of triples based on TransE scores
        
        Args:
            triples: List of triples to evaluate
            
        Returns:
            Dictionary of quality metrics
        """
        scores = []
        
        for h, r, t in triples:
            score = self.model.score_triple(h, r, t)
            scores.append(score)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores)
        }