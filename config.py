"""
Configuration file for Advanced KG Error Correction System
"""
import torch

class Config:
    """System configuration"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data configuration
    # data_path = '/home/kazgu/kazgu/project/claudecode_project/RL_for_kg_correction/start_over/real_kg/WN18RR/'  # Can switch to WN18RR
    data_path = '/data/FB15K237/'  # Can switch to WN18RR
    
    @classmethod
    def get_dataset_name(cls):
        """Extract dataset name from data path"""
        if 'FB15K237' in cls.data_path:
            return 'FB15K237'
        elif 'WN18RR' in cls.data_path:
            return 'WN18RR'
        else:
            return 'Unknown'
    
    # KG Embedding method configuration
    kg_embedding_method = 'transe'  # Options: 'transe', 'rotate'
    
    @classmethod
    def get_kg_model_path(cls):
        """Get KG model path with dataset and method names"""
        dataset = cls.get_dataset_name()
        return f'checkpoints/{dataset}_{cls.kg_embedding_method}_best.pt'
    
    # Legacy paths for backward compatibility
    kg_model_paths = { 
        'transe': 'checkpoints/transe_best.pt',
        'rotate': 'checkpoints/rotate_final.pt'
    }

    # TransE hyperparameters (based on reference implementation)
    class TransE:
        embedding_dim = 128  # Standard dimension from reference
        margin = 16  # gamma value from reference (for gamma - score formulation)
        learning_rate = 0.0005  # Conservative learning rate
        batch_size = 2048  # Standard batch size
        num_epochs = 600  # Sufficient epochs
        negative_samples = 5  # Number of negative samples from reference
        norm = 1  # L1 norm as in reference
        regularization_weight = 0.0  # No weight decay
        epsilon = 2.0  # For embedding range calculation
    
    # RotatE hyperparameters
    class RotatE:
        embedding_dim = 512  # Dimension per complex component
        margin = 9.0  # Margin for RotatE (original paper uses 6.0-12.0)
        learning_rate = 0.0005  # Slightly higher learning rate
        batch_size = 1024
        num_epochs = 300
        epsilon = 2.0  # For initialization
        
    # DQN hyperparameters
    class DQN:
        # Network architecture
        hidden_dims = [512, 256, 128]  # Original architecture
        dropout_rate = 0.2  # Moderate dropout
        
        # Training parameters
        learning_rate = 0.0001  # Conservative learning rate
        batch_size = 2048  # Standard batch size
        gamma = 0.99  # Standard discount factor
        epsilon_start = 1.0
        epsilon_end = 0.01  # Standard minimum exploration
        epsilon_decay = 0.995  # Standard decay rate
        
        # Memory and target network
        memory_size = 10000  # Standard size
        target_update_freq = 100  # Update target network every N steps
        
        # Training episodes
        num_episodes = 200
        max_steps_per_episode = 2000
        
    # Environment configuration
    class Environment:
        # Label-based reward settings (for supervised training)
        reward_accept_correct = 20.0      # True Negative: Accept correct triple
        reward_accept_error = -25.0       # False Negative: Accept error triple
        reward_fix_error_base = 12.0      # True Positive: Fix error (base reward)
        reward_fix_correct = -30.0        # False Positive: Fix correct triple
        reward_correct_error_type = 3.0   # Bonus for identifying correct error type
        reward_perfect_fix = 5.0          # Bonus for perfect fix (restore original)
        
        # Score-based reward settings (for unsupervised training)
        reward_accept_good_score = 5.0    # Accept triple with good TransE score
        reward_accept_bad_score = -5.0    # Accept triple with bad TransE score
        reward_score_improvement = 10.0   # Base reward for improving score
        reward_score_degradation = -10.0  # Penalty for worsening score
        reward_no_change_on_fix = -2.0    # Penalty for fix attempt with no change
        
        # TransE score thresholds (adjusted based on FB15K237 statistics)
        good_triple_threshold = 15.0   # Score below this is considered good (increased)
        bad_triple_threshold = 20.0   # Score above this is considered bad (increased)
        
        # Action bias - encourage acceptance when score is reasonable
        acceptance_bonus_factor = 2.0  # Bonus multiplier for accepting good triples
        
    # Evaluation configuration
    class Evaluation:
        top_k = [1, 3, 10]  # Top-k for prediction accuracy
        eval_batch_size = 256
        
    # Logging configuration
    log_interval = 10  # Log every N episodes/batches
    save_interval = 100  # Save model every N episodes
    model_save_path = './checkpoints/'
    log_path = './logs/'
    
    @classmethod
    def get_dqn_model_path(cls, stage: str = None, best: bool = False):
        """Get DQN model path with dataset and KG method names"""
        dataset = cls.get_dataset_name()
        kg_method = cls.kg_embedding_method
        
        if stage:
            if best:
                return f'{cls.model_save_path}{dataset}_{kg_method}_dqn_{stage}_best.pt'
            else:
                return f'{cls.model_save_path}{dataset}_{kg_method}_dqn_{stage}.pt'
        else:
            if best:
                return f'{cls.model_save_path}{dataset}_{kg_method}_dqn_staged_best.pt'
            else:
                return f'{cls.model_save_path}{dataset}_{kg_method}_dqn.pt'
    
    # Seed for reproducibility
    seed = 42
    
    # Staged training configuration
    class StagedTraining:
        # Stage 1: Full supervision with labels
        stage1_episodes = 160
        stage1_label_weight = 1.0
        
        # Stage 2: Reduced supervision
        stage2_episodes = 160
        stage2_label_weight = 0.5
        
        # Stage 3: No supervision (only TransE features)
        stage3_episodes = 120
        stage3_label_weight = 0.0
        
        # Curriculum learning
        use_curriculum = True
        easy_error_rate = 0.50  # Start with 5% errors
        hard_error_rate = 0.50  # End with 20% errors
    
    # Simple Embedding Adapter configuration
    class EmbeddingAdapter:
        use_adapter = False  # 简单开关 - 是否使用embedding adapter
        output_dim = 128   # 统一输出维度（标准化维度）
        dropout = 0.1      # Dropout率
        
    @classmethod
    def get_state_dim(cls, use_labels=False):
        """Calculate state dimension for DQN - uniform across all KG embedding methods"""
        # State: [h_emb, r_emb, t_emb, score, confidence_features]
        # Labels are NEVER in the state, only used for rewards
        # All embeddings are standardized through adapter (if enabled) or truncation
        
        if cls.EmbeddingAdapter.use_adapter:
            emb_dim = cls.EmbeddingAdapter.output_dim  # Use adapter output dimension
        else:
            emb_dim = 128  # Fallback to truncation size
        
        return 3 * emb_dim + 10  # 3 embeddings + 10 additional features
    
    @classmethod
    def get_action_dim(cls):
        """Get action dimension"""
        return 4  # [Accept, Fix_Head, Fix_Relation, Fix_Tail]