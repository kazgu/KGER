"""
Improved staged DQN training with validation-based model selection
This version uses validation data to select and save the best model
"""
import torch
import os
import json
import random
import tempfile
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from config import Config
from models.transe import TransE
from models.rotate import RotatE
from models.agent import DQNAgent
from utils.data_loader import KGDataLoader
from utils.labeled_kg_dataset import LabeledKGDataset
from environment.labeled_kg_env import LabeledKGEnvironment


def evaluate_on_validation(agent: DQNAgent, kg_model, 
                          data_loader: KGDataLoader, num_samples: int = 1000,
                          error_rate: float = 0.15, use_labels: bool = None,label_weight=0) -> Dict:
    """
    Evaluate agent on validation data
    
    Args:
        agent: DQN agent to evaluate
        kg_model: KG embedding model (TransE or RotatE)
        data_loader: Data loader with validation set
        num_samples: Number of validation samples to use
        error_rate: Error rate for validation data
        use_labels: Whether to use labels (auto-detect if None)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Sample validation data
    val_samples = random.sample(data_loader.valid_triples, 
                               min(num_samples, len(data_loader.valid_triples)))
    
    # Use the improved evaluation function from base_experiment
    from experiments.base_experiment import BaseExperiment
    
    # Create temporary validation dataset
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        val_data = []
        for h, r, t in val_samples:
            h_name = data_loader.id2entity.get(h, str(h))
            r_name = data_loader.id2relation.get(r, str(r))
            t_name = data_loader.id2entity.get(t, str(t))
            val_data.append({'triplet': [h_name, r_name, t_name]})
        json.dump(val_data, f)
        temp_file = f.name
    
    # Create labeled validation dataset with different seed to avoid overfitting
    val_dataset = LabeledKGDataset(
        temp_file,
        data_loader.entity2id,
        data_loader.relation2id,
        noise_rate=error_rate,
        noise_type='mixed',
        seed=random.randint(0, 10000)  # Random seed to avoid overfitting
    )
    
    # Create environment with appropriate label settings
    # State representation is now always the same (no labels in state)
    # Labels are only used for reward calculation via label_weight
    env = LabeledKGEnvironment(
        val_dataset,
        kg_model,
        data_loader,
        use_labels=use_labels if use_labels is not None else True,
        label_weight=label_weight  # Pass the actual label_weight for reward calculation
    )
    


    # Evaluate
    raw_state = env.reset()
    # Get state representation directly from agent
    if env.current_idx < len(env.triples):
        h, r, t = env.triples[env.current_idx]
        score = env.kg_model.score_triple(h, r, t)
        transE_features = env._extract_transE_features(h, r, t, score)
        additional_features = torch.cat([
            torch.tensor([score], device=Config.device),
            transE_features
        ])
        state = agent.get_state_representation(h, r, t, additional_features, training=False)
    else:
        # Fallback to raw state if no valid triple
        state = raw_state
        
    total_reward = 0
    
    with torch.no_grad():
        while True:
            # Use greedy policy for evaluation
            action = agent.select_action(state, training=False)
            raw_next_state, reward, done, _ = env.step(action)
            
            # Process next state through agent
            if not done and env.current_idx < len(env.triples):
                h, r, t = env.triples[env.current_idx]
                score = env.kg_model.score_triple(h, r, t)
                transE_features = env._extract_transE_features(h, r, t, score)
                additional_features = torch.cat([
                    torch.tensor([score], device=Config.device),
                    transE_features
                ])
                next_state = agent.get_state_representation(h, r, t, additional_features, training=False)
            else:
                next_state = torch.zeros(agent.state_dim, device=Config.device) if done else raw_next_state
                
            total_reward += reward
            state = next_state
            
            if done:
                break
    
    # Get statistics
    stats = env.get_statistics()
    
    # Clean up
    os.remove(temp_file)
    
    return {
        'total_reward': total_reward,
        'accuracy': stats['accuracy'],
        'precision': stats['precision'],
        'recall': stats['recall'],
        'f1_score': stats['f1_score'],
        'fix_accuracy': stats.get('fix_accuracy', 0.0),
        'correction_rate': stats.get('correction_rate', 0.0)
    }


def train_stage_with_validation(agent: DQNAgent, kg_model, 
                               data_loader: KGDataLoader,
                               num_episodes: int, label_weight: float,
                               error_rate: float, stage_name: str,
                               start_episode: int = 0,
                               eval_freq: int = 10,
                               patience: int = 20) -> Tuple[Dict, Dict]:
    """
    Train one stage with validation-based early stopping and best model tracking
    
    Args:
        agent: DQN agent
        kg_model: KG embedding model (TransE or RotatE)
        data_loader: Data loader
        num_episodes: Number of training episodes
        label_weight: Weight for label supervision
        error_rate: Error rate for training data
        stage_name: Name of the stage
        start_episode: Starting episode number
        eval_freq: Frequency of validation evaluation
        patience: Patience for early stopping
        
    Returns:
        Tuple of (training_stats, best_model_info)
    """
    # Training statistics
    episode_rewards = []
    episode_accuracies = []
    episode_precisions = []
    episode_recalls = []
    episode_f1_scores = []
    episode_dqn_losses = []

    
    # Validation tracking
    val_f1_scores = []
    best_val_f1 = -float('inf')
    best_episode = 0
    best_model_state = None
    episodes_without_improvement = 0
    
    # Pre-create training data samples to avoid repeated file I/O
    all_training_samples = []
    for _ in range(num_episodes):
        sample_data = random.sample(data_loader.train_triples, 
                                  min(1000, len(data_loader.train_triples)))
        train_data = []
        for h, r, t in sample_data:
            h_name = data_loader.id2entity.get(h, str(h))
            r_name = data_loader.id2relation.get(r, str(r))
            t_name = data_loader.id2entity.get(t, str(t))
            train_data.append({'triplet': [h_name, r_name, t_name]})
        all_training_samples.append(train_data)
    
    for episode in range(num_episodes):
        # Use pre-created training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(all_training_samples[episode], f)
            temp_file = f.name
        
        # Create labeled dataset
        labeled_dataset = LabeledKGDataset(
            temp_file,
            data_loader.entity2id,
            data_loader.relation2id,
            noise_rate=error_rate,
            noise_type='mixed',
            seed=Config.seed
        )
        
        # Create environment
        env = LabeledKGEnvironment(
            labeled_dataset,
            kg_model,
            data_loader,
            use_labels=True,
            label_weight=label_weight
        )
        
        # Run training episode
        raw_state = env.reset()
        # Get state representation from agent
        if env.current_idx < len(env.triples):
            h, r, t = env.triples[env.current_idx]
            score = env.kg_model.score_triple(h, r, t)
            transE_features = env._extract_transE_features(h, r, t, score)
            additional_features = torch.cat([
                torch.tensor([score], device=Config.device),
                transE_features
            ])
            state = agent.get_state_representation(h, r, t, additional_features, training=True)
        else:
            # Fallback to raw state if no valid triple
            state = raw_state
            
        total_reward = 0
        episode_dqn_loss_values = []


        while True:
            action = agent.select_action(state, training=True)
            raw_next_state, reward, done, info = env.step(action)

            # Process next state through agent
            if not done and env.current_idx < len(env.triples):
                h, r, t = env.triples[env.current_idx]
                score = env.kg_model.score_triple(h, r, t)
                transE_features = env._extract_transE_features(h, r, t, score)
                additional_features = torch.cat([
                    torch.tensor([score], device=Config.device),
                    transE_features
                ])
                next_state = agent.get_state_representation(h, r, t, additional_features, training=True)
            else:
                next_state = torch.zeros(agent.state_dim, device=Config.device) if done else raw_next_state

            agent.store_transition(state, action, reward, next_state, done)

            loss_tuple = agent.train_step(batch_size=Config.DQN.batch_size)
            if loss_tuple is not None:
                dqn_loss = loss_tuple
                episode_dqn_loss_values.append(dqn_loss)

            total_reward += reward
            state = next_state

            if done:
                break
        
        # Update epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Get training statistics
        train_stats = env.get_statistics()
        episode_rewards.append(total_reward)
        episode_accuracies.append(train_stats['accuracy'])
        episode_precisions.append(train_stats['precision'])
        episode_recalls.append(train_stats['recall'])
        episode_f1_scores.append(train_stats['f1_score'])

        if episode_dqn_loss_values:
            episode_dqn_losses.append(sum(episode_dqn_loss_values) / len(episode_dqn_loss_values))
        else:
            episode_dqn_losses.append(0.0)
        
        # Clean up training temp file
        os.remove(temp_file)
        
        # Validation evaluation
        if (episode + 1) % eval_freq == 0:
            # Pass the same label_weight as training for consistent reward calculation
            val_stats = evaluate_on_validation(
                agent, kg_model, data_loader,
                num_samples=1000, error_rate=error_rate, 
                use_labels=True,  # Always use labels for evaluation
                label_weight=label_weight  # Same weight as training
            )
            val_f1 = val_stats['f1_score']
            val_f1_scores.append(val_f1)
            
            avg_dqn_loss = sum(episode_dqn_loss_values) / len(episode_dqn_loss_values) if episode_dqn_loss_values else 0.0

            print(f"\n{stage_name} Episode {episode + 1}/{num_episodes}")
            print(f"  Training - Reward: {total_reward:.2f}, F1: {train_stats['f1_score']:.3f}, DQN Loss: {avg_dqn_loss:.4f}")
            print(f"  Validation - F1: {val_f1:.3f}, Fix: {val_stats.get('fix_accuracy', 0):.3f}, Corr: {val_stats.get('correction_rate', 0):.3f} (Best F1: {best_val_f1:.3f})")
            
            # Check if this is the best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_episode = episode + 1
                best_model_state = {
                    'q_network_state': agent.q_network.state_dict(),
                    'target_network_state': agent.target_network.state_dict(),
                    'optimizer_state': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'steps_done': agent.steps_done,
                    'episodes_done': agent.episodes_done
                }
                episodes_without_improvement = 0
                print(f"  ✓ New best model! Validation F1: {best_val_f1:.3f}")
            else:
                episodes_without_improvement += eval_freq
            
            # Early stopping check
            if episodes_without_improvement >= patience:
                print(f"\n⚠ Early stopping triggered after {episode + 1} episodes")
                print(f"  No improvement for {episodes_without_improvement} episodes")
                break
        
        # Print regular progress
        elif (episode + 1) % 10 == 0:
            avg_dqn_loss = sum(episode_dqn_loss_values) / len(episode_dqn_loss_values) if episode_dqn_loss_values else 0.0

            print(f"\n{stage_name} Episode {episode + 1}/{num_episodes}")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Accuracy: {train_stats['accuracy']:.3f}")
            print(f"  F1-Score: {train_stats['f1_score']:.3f}")
            print(f"  DQN Loss: {avg_dqn_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")

    best_path = Config.get_dqn_model_path(stage=stage_name, best=True)
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    torch.save(best_model_state, best_path)
    print(f"✓ Best model saved to: {best_path}")
    # Return statistics and best model info
    return {
        'rewards': episode_rewards,
        'accuracies': episode_accuracies,
        'precisions': episode_precisions,
        'recalls': episode_recalls,
        'f1_scores': episode_f1_scores,
        'dqn_losses': episode_dqn_losses,
        'val_f1_scores': val_f1_scores
    }, {
        'best_val_f1': best_val_f1,
        'best_episode': best_episode,
        'best_model_state': best_model_state
    }


def train_staged_dqn_improved(kg_model_path: str = None):
    """
    Improved staged DQN training with validation-based model selection
    """
    print("=" * 80)
    print("IMPROVED STAGED DQN TRAINING WITH VALIDATION")
    print(f"KG Embedding Method: {Config.kg_embedding_method.upper()}")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = KGDataLoader(Config.data_path)
    print(f"   Entities: {data_loader.get_num_entities()}")
    print(f"   Relations: {data_loader.get_num_relations()}")
    print(f"   Train: {len(data_loader.train_triples)}")
    print(f"   Valid: {len(data_loader.valid_triples)} ← Will be used for model selection")
    print(f"   Test: {len(data_loader.test_triples)}")
    
    # Load KG embedding model based on config
    print(f"\n2. Loading {Config.kg_embedding_method.upper()} model...")
    
    if Config.kg_embedding_method == 'transe':
        kg_model = TransE(
            num_entities=data_loader.get_num_entities(),
            num_relations=data_loader.get_num_relations(),
            embedding_dim=Config.TransE.embedding_dim,
            margin=Config.TransE.margin,
            norm=Config.TransE.norm,
            device=Config.device
        )
    elif Config.kg_embedding_method == 'rotate':
        kg_model = RotatE(
            num_entities=data_loader.get_num_entities(),
            num_relations=data_loader.get_num_relations(),
            embedding_dim=Config.RotatE.embedding_dim,
            margin=Config.RotatE.margin,
            epsilon=Config.RotatE.epsilon,
            device=Config.device
        )
    else:
        raise ValueError(f"Unknown KG embedding method: {Config.kg_embedding_method}")
    
    # Use provided path or get from config with dataset name
    if kg_model_path is None:
        kg_model_path = Config.get_kg_model_path()
        # Fallback to legacy path if new path doesn't exist
        if not os.path.exists(kg_model_path):
            kg_model_path = Config.kg_model_paths[Config.kg_embedding_method]
    
    if os.path.exists(kg_model_path):
        kg_model.load(kg_model_path)
        print(f"   Loaded from: {kg_model_path}")
    else:
        print(f"   Warning: Model file not found at {kg_model_path}")
        raise FileNotFoundError(f"No KG embedding model found at {kg_model_path}")
    
    # Create DQN agent with uniform state dimension (regardless of KG model type)
    state_dim = Config.get_state_dim(use_labels=False)
    print(f"   Using uniform state_dim={state_dim} for all KG models")
    
    action_dim = Config.get_action_dim()
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        kg_model=kg_model,  
        use_dueling=True,
        device=Config.device
    )
    print(f"\n3. Created DQN agent")
    print(f"   State dimension: {state_dim}")
    print(f"   Action dimension: {action_dim}")


    print(f"\n4. Standard DQN training (using {Config.kg_embedding_method.upper()})")
    print(f"   Using standardized embeddings (128D) for consistent DQN input")

    # Track best model across all stages
    global_best_val_f1 = -float('inf')
    global_best_model_state = None
    global_best_stage = ""
    global_best_episode = 0
    
    # Training statistics
    all_stats = {}
    
    # === STAGE 1: Full Supervision ===
    print("\n" + "=" * 60)
    print("STAGE 1: Full Label Supervision")
    print(f"Episodes: {Config.StagedTraining.stage1_episodes}")
    print(f"Label weight: {Config.StagedTraining.stage1_label_weight}")
    print("=" * 60)
    
    stage1_stats, stage1_best = train_stage_with_validation(
        agent, kg_model, data_loader,
        num_episodes=Config.StagedTraining.stage1_episodes,
        label_weight=Config.StagedTraining.stage1_label_weight,
        error_rate=Config.StagedTraining.easy_error_rate,
        stage_name="stage1",
        eval_freq=10,
        patience=100
    )
    
    all_stats['stage1'] = stage1_stats
    
    if stage1_best['best_val_f1'] > global_best_val_f1:
        global_best_val_f1 = stage1_best['best_val_f1']
        global_best_model_state = stage1_best['best_model_state']
        global_best_stage = "stage1"
        global_best_episode = stage1_best['best_episode']


    # === STAGE 2: Reduced Supervision ===
    print("\n" + "=" * 60)
    print("STAGE 2: Reduced Label Supervision")
    print(f"Episodes: {Config.StagedTraining.stage2_episodes}")
    print(f"Label weight: {Config.StagedTraining.stage2_label_weight}")
    print("=" * 60)


    
    stage2_stats, stage2_best = train_stage_with_validation(
        agent, kg_model, data_loader,
        num_episodes=Config.StagedTraining.stage2_episodes,
        label_weight=Config.StagedTraining.stage2_label_weight,
        error_rate=(Config.StagedTraining.easy_error_rate + Config.StagedTraining.hard_error_rate) / 2,
        stage_name="stage2",
        eval_freq=10,
        patience=100
    )
    
    all_stats['stage2'] = stage2_stats
    
    if stage2_best['best_val_f1'] > global_best_val_f1:
        global_best_val_f1 = stage2_best['best_val_f1']
        global_best_model_state = stage2_best['best_model_state']
        global_best_stage = "stage2"
        global_best_episode = Config.StagedTraining.stage1_episodes + stage2_best['best_episode']
    
    # === STAGE 3: No Supervision ===
    print("\n" + "=" * 60)
    print(f"STAGE 3: No Label Supervision (KG embeddings only)")
    print(f"Episodes: {Config.StagedTraining.stage3_episodes}")
    print(f"Label weight: {Config.StagedTraining.stage3_label_weight}")
    print("=" * 60)
    
    stage3_stats, stage3_best = train_stage_with_validation(
        agent, kg_model, data_loader,
        num_episodes=Config.StagedTraining.stage3_episodes,
        label_weight=Config.StagedTraining.stage3_label_weight,
        error_rate=Config.StagedTraining.hard_error_rate,
        stage_name="stage3",
        eval_freq=10,
        patience=100
    )
    
    all_stats['stage3'] = stage3_stats
    
    if stage3_best['best_val_f1'] > global_best_val_f1:
        global_best_val_f1 = stage3_best['best_val_f1']
        global_best_model_state = stage3_best['best_model_state']
        global_best_stage = "stage3"
        global_best_episode = (Config.StagedTraining.stage1_episodes + 
                               Config.StagedTraining.stage2_episodes + 
                               stage3_best['best_episode'])
    
    # === SAVE BEST MODEL ===
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model from: {global_best_stage}, Episode {global_best_episode}")
    print(f"Best validation F1: {global_best_val_f1:.3f}")
    
    # Save best model
    os.makedirs(Config.model_save_path, exist_ok=True)
    
    # Save best model state with dataset and KG method names
    best_path = Config.get_dqn_model_path(best=True)
    torch.save(global_best_model_state, best_path)
    print(f"✓ Best model saved to: {best_path}")
    
    # Also save current model for comparison
    final_path = Config.get_dqn_model_path()
    agent.save(final_path)
    print(f"✓ Final model saved to: {final_path}")
    
    # Restore best model to agent for final evaluation
    agent.q_network.load_state_dict(global_best_model_state['q_network_state'])
    agent.target_network.load_state_dict(global_best_model_state['target_network_state'])
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Final test evaluation with labels for accurate metrics
    test_stats = evaluate_on_validation(
        agent, kg_model, data_loader,
        num_samples=min(1000, len(data_loader.test_triples)),
        error_rate=0.30,
        use_labels=True,  # Use labels for accurate evaluation
        label_weight=1.0  # Full label weight for final evaluation
    )
    
    print(f"Test Accuracy: {test_stats['accuracy']:.3f}")
    print(f"Test Precision: {test_stats['precision']:.3f}")
    print(f"Test Recall: {test_stats['recall']:.3f}")
    print(f"Test F1-Score: {test_stats['f1_score']:.3f}")
    print(f"Test Fix Accuracy: {test_stats.get('fix_accuracy', 0):.3f}")
    print(f"Test Correction Rate: {test_stats.get('correction_rate', 0):.3f}")
    
    # Plot training curves
    plot_training_with_validation(all_stats)
    
    # Save training log
    log_path = f"{Config.log_path}/training_log_improved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(Config.log_path, exist_ok=True)
    
    with open(log_path, 'w') as f:
        log_data = {
            'best_model': {
                'stage': global_best_stage,
                'episode': global_best_episode,
                'val_f1': global_best_val_f1
            },
            'test_results': test_stats,
            'training_stats': all_stats  # Keep all stats including val_f1_scores for analysis
        }
        json.dump(log_data, f, indent=2)
    
    print(f"\n✓ Training log saved to: {log_path}")
    
    return agent


def plot_training_with_validation(all_stats: Dict):
    """Plot training curves with validation performance"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Combine all stages
    all_rewards = []
    all_f1_scores = []
    all_dqn_losses = []
    all_val_f1_scores = []

    for stage in ['stage1', 'stage2', 'stage3']:
        if stage in all_stats:
            all_rewards.extend(all_stats[stage]['rewards'])
            all_f1_scores.extend(all_stats[stage]['f1_scores'])
            if 'dqn_losses' in all_stats[stage]:
                all_dqn_losses.extend(all_stats[stage]['dqn_losses'])
            if 'val_f1_scores' in all_stats[stage]:
                all_val_f1_scores.extend(all_stats[stage]['val_f1_scores'])
    
    episodes = range(1, len(all_rewards) + 1)
    
    # Plot rewards
    axes[0, 0].plot(episodes, all_rewards)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot training F1
    axes[0, 1].plot(episodes, all_f1_scores, label='Training')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Training F1 Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot validation F1
    if all_val_f1_scores:
        val_episodes = list(range(10, len(all_rewards) + 1, 10))[:len(all_val_f1_scores)]
        axes[0, 2].plot(val_episodes, all_val_f1_scores, 'r-o', label='Validation')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('Validation F1 Score')
        axes[0, 2].grid(True, alpha=0.3)

    # Plot DQN loss
    if all_dqn_losses:
        axes[1, 0].plot(episodes, all_dqn_losses, 'g-', label='DQN Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('DQN Loss')
        axes[1, 0].set_title('DQN Training Loss')
        axes[1, 0].set_yscale('log')  # Log scale for loss
        axes[1, 0].grid(True, alpha=0.3)





    # Clear unused subplots
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')

    plt.suptitle('Improved Staged DQN Training with Validation and Loss Tracking', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{Config.log_path}/training_curves_improved_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved staged DQN training')
    parser.add_argument('--kg_model_path', type=str, 
                       default=None,
                       help='Path to KG embedding model (TransE or RotatE)')
    parser.add_argument('--kg_method', type=str,
                       default=None,
                       help='Override KG embedding method (transe or rotate)')
    
    args = parser.parse_args()
    
    # Override config if method specified
    if args.kg_method:
        Config.kg_embedding_method = args.kg_method.lower()
    
    # Train with improved staged supervision
    agent = train_staged_dqn_improved(kg_model_path=args.kg_model_path)