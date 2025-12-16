"""
Knowledge Graph Environment for Reinforcement Learning
"""
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from models.transe import TransE
from config import Config


@dataclass
class EnvState:
    """Environment state"""
    triple_idx: int
    current_triple: Tuple[int, int, int]
    embeddings: torch.Tensor
    score: float
    done: bool


class KGEnvironment:
    """
    Knowledge Graph Environment for RL-based error detection and correction
    """
    
    def __init__(self, triples: List[Tuple[int, int, int]], 
                 transe_model: TransE, 
                 data_loader=None):
        """
        Initialize KG environment
        
        Args:
            triples: List of KG triples to process
            transe_model: Pre-trained TransE model
            data_loader: Data loader for ID-name mappings
        """
        self.triples = triples
        self.transe_model = transe_model
        self.data_loader = data_loader
        
        # Freeze TransE model
        for param in self.transe_model.parameters():
            param.requires_grad = False
        
        self.current_idx = 0
        self.max_steps = len(triples)
        self.modified_triples = []
        self.action_history = []
        self.modified_indices = []  # Track which indices were modified
        
        # Action space: 0=Accept, 1=Fix_Head, 2=Fix_Relation, 3=Fix_Tail
        self.action_space = 4
        
        # Statistics
        self.stats = {
            'total_triples': len(triples),
            'accepted': 0,
            'modified': 0,
            'score_improvements': [],
            'action_counts': {0: 0, 1: 0, 2: 0, 3: 0}
        }
        
    def reset(self) -> torch.Tensor:
        """
        Reset environment to initial state
        
        Returns:
            Initial state
        """
        self.current_idx = 0
        self.modified_triples = []
        self.action_history = []
        self.modified_indices = []
        
        # Reset statistics
        self.stats['accepted'] = 0
        self.stats['modified'] = 0
        self.stats['score_improvements'] = []
        self.stats['action_counts'] = {0: 0, 1: 0, 2: 0, 3: 0}
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """
        Get current state representation
        
        Returns:
            State tensor
        """
        if self.current_idx >= len(self.triples):
            # Return zero state if done
            return torch.zeros(Config.get_state_dim(), device=Config.device)
        
        h, r, t = self.triples[self.current_idx]
        
        # Get embeddings
        h_emb, r_emb, t_emb = self.transe_model.get_embeddings(h, r, t)
        
        # Calculate TransE score
        score = self.transe_model.score_triple(h, r, t)
        
        # Additional features
        additional_features = self._extract_additional_features(h, r, t, score)
        
        # Concatenate all features
        state = torch.cat([
            h_emb,
            r_emb,
            t_emb,
            torch.tensor([score], device=Config.device),
            additional_features
        ])
        
        # Pad or truncate to match expected state dimension
        expected_dim = Config.get_state_dim()
        if state.shape[0] < expected_dim:
            padding = torch.zeros(expected_dim - state.shape[0], device=Config.device)
            state = torch.cat([state, padding])
        elif state.shape[0] > expected_dim:
            state = state[:expected_dim]
        
        return state
    
    def _extract_additional_features(self, h: int, r: int, t: int, 
                                    score: float) -> torch.Tensor:
        """
        Extract additional features for state representation
        
        Args:
            h: Head entity ID
            r: Relation ID
            t: Tail entity ID
            score: TransE score
            
        Returns:
            Additional feature tensor
        """
        features = []
        
        # Score-based features (more granular)
        features.append(np.exp(-score/5.0))  # Exponential decay of score (good triples have higher values)
        features.append(min(score / 20.0, 1.0))  # Normalized score (scaled differently)
        features.append(1.0 if score < 3.0 else 0.0)  # Very good triple
        features.append(1.0 if score < 6.0 else 0.0)  # Good triple
        features.append(1.0 if score < 9.0 else 0.0)  # Acceptable triple
        features.append(1.0 if score > 12.0 else 0.0)  # Bad triple
        
        # Ranking features (check top-k instead of just top-1)
        pred_tails, tail_scores = self.transe_model.predict_tail(h, r, k=10)
        pred_heads, head_scores = self.transe_model.predict_head(r, t, k=10)
        pred_rels, rel_scores = self.transe_model.predict_relation(h, t, k=10)
        
        # Tail ranking feature (smoother)
        if t in pred_tails:
            tail_rank = pred_tails.index(t) + 1
            features.append(1.0 / tail_rank)  # Reciprocal rank
        else:
            features.append(0.0)
        
        # Head ranking feature
        if h in pred_heads:
            head_rank = pred_heads.index(h) + 1
            features.append(1.0 / head_rank)
        else:
            features.append(0.0)
        
        # Relation ranking feature
        if r in pred_rels:
            rel_rank = pred_rels.index(r) + 1
            features.append(1.0 / rel_rank)
        else:
            features.append(0.0)
        
        return torch.tensor(features, device=Config.device, dtype=torch.float32)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Execute action in environment
        
        Args:
            action: Action to execute (0-3)
            
        Returns:
            (next_state, reward, done, info)
        """
        if self.current_idx >= len(self.triples):
            return self._get_state(), 0.0, True, {}
        
        # Get current triple
        old_triple = self.triples[self.current_idx]
        h, r, t = old_triple
        
        # Calculate old score
        old_score = self.transe_model.score_triple(h, r, t)
        
        # Execute action
        new_triple = self._execute_action(action, old_triple)
        new_h, new_r, new_t = new_triple
        
        # Calculate new score
        new_score = self.transe_model.score_triple(new_h, new_r, new_t)


        
        # Calculate reward
        reward = self._calculate_reward(action, old_score, new_score)
        
        # Update statistics
        self.stats['action_counts'][action] += 1
        if action == 0:
            self.stats['accepted'] += 1
        else:
            self.stats['modified'] += 1
            self.stats['score_improvements'].append(old_score - new_score)
            self.modified_indices.append(self.current_idx)  # Track modified index
        
        # Store modified triple
        self.modified_triples.append(new_triple)
        self.action_history.append(action)
        
        # Move to next triple
        self.current_idx += 1
        done = (self.current_idx >= len(self.triples))
        
        # Get next state
        next_state = self._get_state()
        
        # Prepare info
        info = {
            'old_triple': old_triple,
            'new_triple': new_triple,
            'old_score': old_score,
            'new_score': new_score,
            'action': action,
            'improvement': old_score - new_score
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: int, triple: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Execute action on triple
        
        Args:
            action: Action to execute
            triple: Current triple
            
        Returns:
            Modified triple
        """
        h, r, t = triple
        
        if action == 0:  # Accept
            return triple
            
        elif action == 1:  # Fix Head
            # Predict best head given relation and tail
            pred_heads, _ = self.transe_model.predict_head(r, t, k=1)
            return (pred_heads[0], r, t)
            
        elif action == 2:  # Fix Relation
            # Predict best relation given head and tail
            pred_rels, _ = self.transe_model.predict_relation(h, t, k=1)
            return (h, pred_rels[0], t)
            
        elif action == 3:  # Fix Tail
            # Predict best tail given head and relation
            pred_tails, _ = self.transe_model.predict_tail(h, r, k=1)
            return (h, r, pred_tails[0])
            
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _calculate_reward(self, action: int, old_score: float, new_score: float) -> float:
        """
        Calculate reward for action
        
        Args:
            action: Action taken
            old_score: Score before action
            new_score: Score after action
            
        Returns:
            Reward value
        """
        improvement = old_score - new_score
        
        if action == 0:  # Accept
            # Strong reward for accepting good triples
            if old_score < Config.Environment.good_triple_threshold:
                # The better the score, the higher the reward
                quality_bonus = max(0, Config.Environment.good_triple_threshold - old_score)
                return Config.Environment.reward_correct_accept + quality_bonus * Config.Environment.acceptance_bonus_factor
            elif old_score > Config.Environment.bad_triple_threshold:
                # Strong penalty for accepting clearly bad triples
                return -Config.Environment.reward_correct_accept
            else:
                # Mild positive reward for accepting borderline cases (encourage acceptance)
                return 2.0
                
        else:  # Correction actions
            # Check if the original was actually good
            if old_score < Config.Environment.good_triple_threshold:
                # Penalize unnecessary changes to good triples
                if improvement <= 0:
                    return Config.Environment.reward_unnecessary_change
                else:
                    # Small reward only if improvement is significant
                    return improvement * Config.Environment.reward_improvement_factor if improvement > 1.0 else 0
            
            # For bad triples, reward improvements
            if improvement > 0:
                # Reward proportional to improvement
                base_reward = Config.Environment.reward_correct_fix
                improvement_bonus = improvement * Config.Environment.reward_improvement_factor
                return base_reward + improvement_bonus
                
            elif improvement == 0:
                # No improvement
                return Config.Environment.reward_no_improvement
                
            else:
                # Made it worse
                return Config.Environment.reward_worse
    
    def get_modified_kg(self) -> List[Tuple[int, int, int]]:
        """
        Get the modified knowledge graph
        
        Returns:
            List of modified triples
        """
        return self.modified_triples
    
    def get_modified_indices(self) -> List[int]:
        """
        Get the indices of modified triples
        
        Returns:
            List of indices that were modified
        """
        return self.modified_indices
    
    def get_statistics(self) -> Dict:
        """
        Get environment statistics
        
        Returns:
            Statistics dictionary
        """
        if self.stats['score_improvements']:
            avg_improvement = np.mean(self.stats['score_improvements'])
        else:
            avg_improvement = 0.0
            
        self.stats['avg_improvement'] = avg_improvement
        self.stats['modification_rate'] = self.stats['modified'] / max(1, self.stats['accepted'] + self.stats['modified'])
        
        return self.stats
    
    def render(self, num_samples: int = 5):
        """
        Render environment state (print sample corrections)
        
        Args:
            num_samples: Number of samples to show
        """
        print("\n=== Sample Corrections ===")
        
        if not self.data_loader:
            print("No data loader available for rendering")
            return
            
        # Show some corrections
        for i in range(min(num_samples, len(self.action_history))):
            if self.action_history[i] != 0:  # Show only corrections
                old_triple = self.triples[i]
                new_triple = self.modified_triples[i]
                
                old_names = self.data_loader.triple_to_names(old_triple)
                new_names = self.data_loader.triple_to_names(new_triple)
                
                old_score = self.transe_model.score_triple(*old_triple)
                new_score = self.transe_model.score_triple(*new_triple)
                
                action_names = ["Accept", "Fix Head", "Fix Relation", "Fix Tail"]
                
                print(f"\nAction: {action_names[self.action_history[i]]}")
                print(f"Original: {old_names} (score: {old_score:.3f})")
                print(f"Modified: {new_names} (score: {new_score:.3f})")
                print(f"Improvement: {old_score - new_score:.3f}")