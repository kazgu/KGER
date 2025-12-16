"""
Labeled KG Environment with supervision signals
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from config import Config


class LabeledKGEnvironment:
    """
    Environment for KG error correction with label supervision
    """
    
    def __init__(self, labeled_dataset, kg_model, data_loader, 
                 use_labels=True, label_weight=1.0):
        """
        Initialize labeled environment
        
        Args:
            labeled_dataset: LabeledKGDataset instance
            kg_model: Pre-trained KG embedding model (TransE or RotatE)
            data_loader: KGDataLoader instance
            use_labels: Whether to use label information in state
            label_weight: Weight of label features in state (for curriculum learning)
        """
        self.dataset = labeled_dataset
        self.triples = labeled_dataset.triples
        self.labels = labeled_dataset.labels
        self.error_types = labeled_dataset.error_types
        self.original_triples = labeled_dataset.original_triples
        
        self.kg_model = kg_model
        self.data_loader = data_loader
        self.use_labels = use_labels
        self.label_weight = label_weight
        
        # Freeze KG embedding model
        for param in self.kg_model.parameters():
            param.requires_grad = False
        
        self.current_idx = 0
        self.max_steps = len(self.triples)
        self.modified_triples = []
        self.action_history = []
        self.modified_indices = []
        
        # Action space: 0=Accept, 1=Fix_Head, 2=Fix_Relation, 3=Fix_Tail
        self.action_space = 4
        
        # Statistics
        self.stats = {
            'total_triples': len(self.triples),
            'true_positives': 0,  # Correctly identified errors (any fix attempt)
            'false_positives': 0,  # Incorrectly modified correct triples
            'true_negatives': 0,  # Correctly accepted correct triples
            'false_negatives': 0,  # Missed errors
            'correct_fixes': 0,  # Errors fixed with perfect restoration
            'correct_error_type': 0,  # Errors fixed with correct action type
            'action_counts': {0: 0, 1: 0, 2: 0, 3: 0},
            'error_type_accuracy': {'head': 0, 'relation': 0, 'tail': 0}
        }
    
    def reset(self) -> torch.Tensor:
        """Reset environment to initial state"""
        self.current_idx = 0
        self.modified_triples = []
        self.action_history = []
        self.modified_indices = []
        
        # Reset statistics
        self.stats['true_positives'] = 0
        self.stats['false_positives'] = 0
        self.stats['true_negatives'] = 0
        self.stats['false_negatives'] = 0
        self.stats['correct_fixes'] = 0
        self.stats['correct_error_type'] = 0
        self.stats['action_counts'] = {0: 0, 1: 0, 2: 0, 3: 0}
        self.stats['error_type_accuracy'] = {'head': 0, 'relation': 0, 'tail': 0}
        
        return self._get_state()
    
    def _get_state(self) -> torch.Tensor:
        """
        Get current state WITHOUT label information
        Labels should only be used for reward calculation, not state representation
        
        State includes:
        - TransE embeddings 
        - TransE score and features
        """
        if self.current_idx >= len(self.triples):
            # Return zero state if done
            return torch.zeros(self._get_state_dim(), device=Config.device)
        
        h, r, t = self.triples[self.current_idx]
        
        # Get KG embeddings
        h_emb, r_emb, t_emb = self.kg_model.get_embeddings(h, r, t)
        
        # Calculate KG embedding score
        score = self.kg_model.score_triple(h, r, t)
        
        # Get TransE-based features
        transE_features = self._extract_transE_features(h, r, t, score)
        
        # State components - NO LABELS
        state_components = [
            h_emb,
            r_emb,
            t_emb,
            torch.tensor([score], device=Config.device),
            transE_features
        ]
        
        # Concatenate all features
        state = torch.cat(state_components)
        
        return state
    
    def _extract_transE_features(self, h: int, r: int, t: int, score: float) -> torch.Tensor:
        """Extract TransE-based features (must match kg_env for consistency)"""
        features = []
        
        # Score-based features (more granular - matching kg_env)
        features.append(np.exp(-score/5.0))  # Exponential decay of score
        features.append(min(score / 20.0, 1.0))  # Normalized score
        features.append(1.0 if score < 3.0 else 0.0)  # Very good triple
        features.append(1.0 if score < 6.0 else 0.0)  # Good triple
        features.append(1.0 if score < 9.0 else 0.0)  # Acceptable triple
        features.append(1.0 if score > 12.0 else 0.0)  # Bad triple
        
        # Ranking features (check top-k)
        pred_tails, _ = self.kg_model.predict_tail(h, r, k=10)
        pred_heads, _ = self.kg_model.predict_head(r, t, k=10)
        pred_rels, _ = self.kg_model.predict_relation(h, t, k=10)
        
        # Tail ranking feature
        if t in pred_tails:
            features.append(1.0 / (pred_tails.index(t) + 1))
        else:
            features.append(0.0)
        
        # Head ranking feature
        if h in pred_heads:
            features.append(1.0 / (pred_heads.index(h) + 1))
        else:
            features.append(0.0)
        
        # Relation ranking feature
        if r in pred_rels:
            features.append(1.0 / (pred_rels.index(r) + 1))
        else:
            features.append(0.0)
        
        return torch.tensor(features, device=Config.device, dtype=torch.float32)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute action and return (next_state, reward, done, info)"""
        if self.current_idx >= len(self.triples):
            return self._get_state(), 0.0, True, {}
        
        # Get current triple and label
        old_triple = self.triples[self.current_idx]
        label = self.labels[self.current_idx]
        error_type = self.error_types[self.current_idx]
        original_triple = self.original_triples[self.current_idx]
        
        # Execute action
        new_triple = self._execute_action(action, old_triple)
        
        # Calculate reward based on label availability
        if self.use_labels and self.label_weight > 0:
            # Use label-based reward when labels are available
            reward = self._calculate_label_based_reward(
                action, label, error_type, new_triple, original_triple
            )
        else:
            # Use TransE score-based reward when no labels
            reward = self._calculate_score_based_reward(
                action, old_triple, new_triple
            )
        
        # Update statistics
        self._update_statistics(action, label, error_type, new_triple, original_triple)
        
        # Store modified triple and action
        self.modified_triples.append(new_triple)
        self.action_history.append(action)
        if action != 0:
            self.modified_indices.append(self.current_idx)
        
        # Move to next triple
        self.current_idx += 1
        done = (self.current_idx >= len(self.triples))
        
        # Get next state
        next_state = self._get_state()
        
        # Prepare info
        info = {
            'label': label,
            'error_type': error_type,
            'action': action,
            'reward': reward,
            'stats': self.get_statistics()
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: int, triple: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Execute action on triple with self-loop prevention"""
        h, r, t = triple
        
        if action == 0:  # Accept
            return triple
        elif action == 1:  # Fix Head
            pred_heads, _ = self.kg_model.predict_head(r, t, k=10)  # Get more candidates
            if pred_heads:
                # Find first prediction that doesn't create self-loop
                for pred_h in pred_heads:
                    if pred_h != t:  # Avoid self-loop
                        return (pred_h, r, t)
                # Fallback: use first prediction even if it creates self-loop
                return (pred_heads[0], r, t)
            return triple
        elif action == 2:  # Fix Relation
            pred_rels, _ = self.kg_model.predict_relation(h, t, k=10)
            if pred_rels:
                # Relations don't directly cause self-loops, but filter out same relation
                for pred_r in pred_rels:
                    if pred_r != r:  # Use different relation
                        return (h, pred_r, t)
                # Fallback: use first prediction
                return (h, pred_rels[0], t)
            return triple
        elif action == 3:  # Fix Tail
            pred_tails, _ = self.kg_model.predict_tail(h, r, k=10)
            if pred_tails:
                # Find first prediction that doesn't create self-loop
                for pred_t in pred_tails:
                    if pred_t != h:  # Avoid self-loop
                        return (h, r, pred_t)
                # Fallback: use first prediction even if it creates self-loop
                return (h, r, pred_tails[0])
            return triple
        
        return triple
    
    def _calculate_label_based_reward(self, action, label, error_type, 
                                      new_triple, original_triple):
        """
        Calculate reward based on true labels
        NOTE: Statistics are updated in _update_statistics, not here
        
        REBALANCED Reward structure (from Config.Environment):
        - True Negative (Accept correct): +20 (reward_accept_correct)
        - True Positive (Fix error): +12 base (reward_fix_error_base)
        - False Negative (Accept error): -25 (reward_accept_error)
        - False Positive (Modify correct): -30 (reward_fix_correct)
        - Correct error type bonus: +3 (reward_correct_error_type)
        - Perfect fix bonus: +5 (reward_perfect_fix)
        
        Perfect Fix Total: +20 (12+3+5) = Same as Accept Correct (+20)
        This balances the expected rewards and fixes the noise rate paradox
        """
        reward = 0.0
        
        if label == 0:  # Triple is correct
            if action == 0:  # Accept correct → True Negative
                reward = Config.Environment.reward_accept_correct
            else:  # Modify correct → False Positive
                reward = Config.Environment.reward_fix_correct
                
        else:  # Triple has error (label == 1)
            if action == 0:  # Accept error → False Negative
                reward = Config.Environment.reward_accept_error
            else:  # Modify error → True Positive
                # Base reward for attempting to fix error
                reward = Config.Environment.reward_fix_error_base
                
                # Check if error type was correctly identified
                if (error_type == 'head' and action == 1) or \
                   (error_type == 'relation' and action == 2) or \
                   (error_type == 'tail' and action == 3):
                    reward += Config.Environment.reward_correct_error_type
                
                # Check if fix restored original triple
                if new_triple == original_triple:
                    reward += Config.Environment.reward_perfect_fix
        
        return reward
    
    def _calculate_score_based_reward(self, action, old_triple, new_triple):
        """
        Calculate reward based on TransE scores when labels are not available
        
        Reward structure:
        - Accept with good score: +5
        - Accept with bad score: -5
        - Fix that improves score: +10 * improvement_ratio
        - Fix that worsens score: -10
        - No change when fix attempted: -2
        """
        old_h, old_r, old_t = old_triple
        new_h, new_r, new_t = new_triple
        
        old_score = self.kg_model.score_triple(old_h, old_r, old_t)
        
        if action == 0:  # Accept
            # Reward based on score quality using Config thresholds
            if old_score < Config.Environment.good_triple_threshold:  # Good triple
                return Config.Environment.reward_accept_good_score
            elif old_score < Config.Environment.bad_triple_threshold:  # Mediocre triple
                return 0.0
            else:  # Bad triple
                return Config.Environment.reward_accept_bad_score
        else:  # Fix action
            if new_triple == old_triple:
                # No change despite fix action
                return Config.Environment.reward_no_change_on_fix
            
            new_score = self.kg_model.score_triple(new_h, new_r, new_t)
            score_improvement = old_score - new_score  # Lower score is better
            
            if score_improvement > 0:
                # Score improved
                improvement_ratio = min(score_improvement / old_score, 1.0)
                return Config.Environment.reward_score_improvement * improvement_ratio
            else:
                # Score worsened
                return Config.Environment.reward_score_degradation
    
    def _update_statistics(self, action, label, error_type, new_triple, original_triple):
        """Update environment statistics based on true labels"""
        self.stats['action_counts'][action] += 1
        
        # Update confusion matrix statistics based on true labels
        if label == 0:  # Triple is correct
            if action == 0:  # Accept correct → True Negative
                self.stats['true_negatives'] += 1
            else:  # Modify correct → False Positive
                self.stats['false_positives'] += 1
                
        else:  # Triple has error (label == 1)
            if action == 0:  # Accept error → False Negative
                self.stats['false_negatives'] += 1
            else:  # Modify error → True Positive
                self.stats['true_positives'] += 1
                
                # Check if error type was correctly identified
                if (error_type == 'head' and action == 1) or \
                   (error_type == 'relation' and action == 2) or \
                   (error_type == 'tail' and action == 3):
                    self.stats['error_type_accuracy'][error_type] += 1
                    self.stats['correct_error_type'] += 1
                
                # Check if fix restored original triple
                if new_triple == original_triple:
                    self.stats['correct_fixes'] += 1
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        total_processed = max(1, self.current_idx)
        
        # Calculate metrics
        tp = self.stats['true_positives']
        fp = self.stats['false_positives']
        tn = self.stats['true_negatives']
        fn = self.stats['false_negatives']
        
        # Basic classification metrics
        accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_score = 2 * precision * recall / max(1e-8, precision + recall)
        
        # Correction quality metrics
        total_errors = tp + fn  # Total errors in dataset
        total_attempted_fixes = tp  # Errors where fix was attempted
        correct_fixes = self.stats['correct_fixes']
        correct_error_type = self.stats['correct_error_type']
        
        # Fix accuracy: among attempted fixes, how many were perfect
        fix_accuracy = correct_fixes / max(1, total_attempted_fixes)
        
        # Error type accuracy: among attempted fixes, how many identified correct error type  
        error_type_accuracy = correct_error_type / max(1, total_attempted_fixes)
        
        # Overall correction rate: perfect fixes / total errors
        correction_rate = correct_fixes / max(1, total_errors)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fix_accuracy': fix_accuracy,
            'error_type_accuracy_rate': error_type_accuracy,
            'correction_rate': correction_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'correct_fixes': correct_fixes,
            'correct_error_type': correct_error_type,
            'total_errors': total_errors,
            'action_distribution': self.stats['action_counts'],
            'error_type_accuracy': self.stats['error_type_accuracy']
        }
    
    def _get_state_dim(self) -> int:
        """Calculate state dimension - depends on KG embedding method"""
        # Use the centralized state dimension calculation from Config
        # This ensures consistency with the DQN agent's expected input dimension
        return Config.get_state_dim(use_labels=False)
    
    def get_modified_indices(self) -> List[int]:
        """Get indices of modified triples"""
        return self.modified_indices