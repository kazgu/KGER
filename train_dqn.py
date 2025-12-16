"""
Training script for DQN agent
"""
import os
import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from config import Config
from models.transe import TransE
from models.agent import DQNAgent
from environment.kg_env import KGEnvironment
from utils.data_loader import KGDataLoader


def train_dqn(transe_path: str = None, use_sample_data: bool = False,
              noise_rate: float = 0.0, noise_type: str = 'mixed', 
              score_based_noise: bool = False):
    """
    Train DQN agent for KG error correction
    
    Args:
        transe_path: Path to pre-trained TransE model
        use_sample_data: Whether to use sample data
        noise_rate: Percentage of triples to corrupt during training (0.0 to 1.0)
        noise_type: Type of corruption ('relation', 'entity', 'mixed')
        score_based_noise: If True, inject noise based on TransE scores
    """
    print("=== DQN Training for KG Error Correction ===\n")
    if noise_rate > 0:
        noise_strategy = "score-based" if score_based_noise else "random"
        print(f"Noise injection: {noise_rate*100:.0f}% ({noise_type} corruption, {noise_strategy} selection)\n")
    
    # Set random seed
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # Load data

    print(f"Loading dataset from {Config.data_path}...")
    data_loader = KGDataLoader(Config.data_path)
    
    # Load TransE model
    print("\nLoading TransE model...")
    transe_model = TransE(
        num_entities=data_loader.get_num_entities(),
        num_relations=data_loader.get_num_relations(),
        embedding_dim=Config.TransE.embedding_dim,
        margin=Config.TransE.margin,
        norm=Config.TransE.norm,
        device=Config.device
    )
    
    if transe_path and os.path.exists(transe_path):
        print(f"Loading pre-trained TransE from {transe_path}")
        transe_model.load(transe_path)
    else:
        print("Warning: No pre-trained TransE model found. Using random initialization.")
        print("Please run train_transe.py first for better performance.\n")
    
    # Prepare training triples with optional noise
    if noise_rate > 0:
        # Create noisy training data
        from utils.kg_dataset import KGDataset
        import json
        
        # Prepare training triples
        if data_loader.train_triples:
            original_triples = data_loader.train_triples
        else:
            original_triples = data_loader.triples
        
        # Save training triples temporarily
        temp_train_file = '/tmp/train_temp.json'
        train_data = []
        for h, r, t in original_triples:
            h_name = data_loader.id2entity.get(h, str(h))
            r_name = data_loader.id2relation.get(r, str(r))
            t_name = data_loader.id2entity.get(t, str(t))
            train_data.append({'triplet': [h_name, r_name, t_name]})
        
        with open(temp_train_file, 'w') as f:
            json.dump(train_data, f)
        
        # Create noisy dataset
        noisy_dataset = KGDataset(temp_train_file, data_loader.entity2id, data_loader.relation2id,
                                noise_rate=noise_rate, noise_type=noise_type, seed=Config.seed,
                                transe_model=transe_model if score_based_noise else None,
                                score_based_noise=score_based_noise)
        training_triples = noisy_dataset.triples
        
        stats = noisy_dataset.get_corruption_stats()
        print(f"Injected noise to training data: {stats['corrupted_triples']}/{stats['total_triples']} ({stats['corruption_rate']:.1%})")
        
        # Clean up temp file
        os.remove(temp_train_file)
    else:
        training_triples = data_loader.train_triples if data_loader.train_triples else data_loader.triples
    
    # Create environment with potentially noisy data
    env = KGEnvironment(
        triples=training_triples,
        transe_model=transe_model,
        data_loader=data_loader
    )
    
    # Create DQN agent
    state_dim = Config.get_state_dim()
    action_dim = Config.get_action_dim()
    
    print(f"\nInitializing DQN Agent...")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        use_dueling=True,  # Use dueling DQN
        device=Config.device
    )
    
    # Training loop
    num_episodes = Config.DQN.num_episodes
    best_reward = -float('inf')
    
    episode_rewards = []
    episode_improvements = []
    episode_losses = []
    episode_accuracies = []
    episode_f1_scores = []
    
    # Initialize logging
    os.makedirs('logs', exist_ok=True)
    training_log = {
        'model_type': 'DQN-Standard',
        'start_time': datetime.now().isoformat(),
        'config': {
            'state_dim': Config.get_state_dim(),
            'action_dim': Config.get_action_dim(),
            'num_episodes': num_episodes,
            'noise_rate': noise_rate,
            'noise_type': noise_type,
            'score_based_noise': score_based_noise
        },
        'episodes': [],
        'rewards': [],
        'losses': [],
        'accuracies': [],
        'improvements': []
    }
    
    print(f"\nStarting training for {num_episodes} episodes...")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Train one episode
        total_reward, steps = agent.train_episode(env, Config.DQN.max_steps_per_episode)
        
        # Get statistics
        stats = env.get_statistics()
        avg_improvement = stats['avg_improvement']
        
        episode_rewards.append(total_reward)
        episode_improvements.append(avg_improvement)
        
        # Get average loss
        if agent.loss_history:
            recent_losses = agent.loss_history[-100:]
            avg_loss = np.mean(recent_losses)
            episode_losses.append(avg_loss)
        
        # Log progress
        if (episode + 1) % Config.log_interval == 0:
            recent_rewards = episode_rewards[-Config.log_interval:]
            recent_improvements = episode_improvements[-Config.log_interval:]
            
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            print(f"Average Reward: {np.mean(recent_rewards):.2f}")
            print(f"Average Score Improvement: {np.mean(recent_improvements):.4f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Actions taken: {stats['action_counts']}")
            print(f"Modification rate: {stats['modification_rate']:.2%}")
            
            # Show sample corrections
            if (episode + 1) % (Config.log_interval * 5) == 0:
                env.render(num_samples=3)
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            save_path = f"{Config.model_save_path}/dqn_best.pt"
            os.makedirs(Config.model_save_path, exist_ok=True)
            agent.save(save_path)
        
        # Save checkpoint
        if (episode + 1) % Config.save_interval == 0:
            save_path = f"{Config.model_save_path}/dqn_episode_{episode + 1}.pt"
            agent.save(save_path)
            print(f"Checkpoint saved to {save_path}")
    
    # Save final model
    final_path = f"{Config.model_save_path}/dqn_final.pt"
    agent.save(final_path)
    print(f"\nTraining completed! Final model saved to {final_path}")
    
    # Save training log
    training_log['episodes'] = list(range(num_episodes))
    training_log['rewards'] = episode_rewards
    training_log['losses'] = [float(l) if l is not None else 0.0 for l in episode_losses]
    training_log['improvements'] = episode_improvements
    training_log['end_time'] = datetime.now().isoformat()
    
    log_path = 'logs/dqn_training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved to {log_path}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_improvements, episode_losses, agent.epsilon_history)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    evaluate_agent(agent, env, data_loader)
    
    return agent, env


def evaluate_agent(agent: DQNAgent, env: KGEnvironment, data_loader: KGDataLoader):
    """
    Evaluate trained agent
    
    Args:
        agent: Trained DQN agent
        env: Environment
        data_loader: Data loader
    """
    print("\nEvaluating agent performance...")
    
    # Evaluate on multiple episodes
    avg_reward, avg_improvement = agent.evaluate(env, num_episodes=10)
    
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Score Improvement: {avg_improvement:.4f}")
    
    # Get final statistics
    stats = env.get_statistics()
    
    print("\n--- Final Statistics ---")
    print(f"Total triples processed: {stats['total_triples']}")
    print(f"Triples accepted: {stats['accepted']} ({stats['accepted']/stats['total_triples']:.1%})")
    print(f"Triples modified: {stats['modified']} ({stats['modified']/stats['total_triples']:.1%})")
    print(f"Action distribution: {stats['action_counts']}")
    
    # Show sample corrections
    print("\n--- Sample Corrections ---")
    env.render(num_samples=10)
    
    # Calculate overall KG quality improvement
    original_scores = []
    modified_scores = []
    
    for i in range(min(100, len(env.triples))):
        original = env.triples[i]
        modified = env.modified_triples[i] if i < len(env.modified_triples) else original
        
        original_score = env.transe_model.score_triple(*original)
        modified_score = env.transe_model.score_triple(*modified)

        print('original_score',original_score)
        print('modified_score',modified_score)
        
        original_scores.append(original_score)
        modified_scores.append(modified_score)
    
    avg_original = np.mean(original_scores)
    avg_modified = np.mean(modified_scores)
    
    print(f"\n--- KG Quality Improvement ---")
    print(f"Average original TransE score: {avg_original:.4f}")
    print(f"Average modified TransE score: {avg_modified:.4f}")
    print(f"Overall improvement: {(avg_original - avg_modified):.4f} ({(avg_original - avg_modified)/avg_original:.1%})")


def plot_training_curves(rewards, improvements, losses, epsilons):
    """
    Plot training curves
    
    Args:
        rewards: Episode rewards
        improvements: Score improvements
        losses: Training losses
        epsilons: Epsilon values
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].grid(True)
    
    # Plot improvements
    axes[0, 1].plot(improvements)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Score Improvement')
    axes[0, 1].set_title('TransE Score Improvements')
    axes[0, 1].grid(True)
    
    # Plot losses
    if losses:
        axes[1, 0].plot(losses)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True)
    
    # Plot epsilon
    axes[1, 1].plot(epsilons)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    os.makedirs(Config.log_path, exist_ok=True)
    plt.savefig(f"{Config.log_path}/dqn_training_curves.png")
    plt.close()
    
    print(f"Training curves saved to {Config.log_path}/dqn_training_curves.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for KG error correction')
    parser.add_argument('--noise_rate', type=float, default=0.2,
                       help='Noise rate for training data (0.0 to 1.0)')
    parser.add_argument('--noise_type', type=str, default='mixed',
                       choices=['relation', 'entity', 'mixed'],
                       help='Type of noise injection')
    parser.add_argument('--score_based', action='store_true',
                       help='Use score-based noise injection (select worst scoring triples)')
    parser.add_argument('--transe_path', type=str, default='checkpoints/transe_final.pt',
                       help='Path to TransE model')
    
    args = parser.parse_args()
    
    # Check for TransE model if not specified
    if args.transe_path is None:
        transe_path = f"{Config.model_save_path}/transe_final.pt"
        if not os.path.exists(transe_path):
            print("Warning: TransE model not found. Training with random initialization.")
            print("Run 'python train_transe.py' first for better performance.\n")
            transe_path = None
        else:
            args.transe_path = transe_path
    
    # Train DQN
    agent, env = train_dqn(
        transe_path=args.transe_path, 
        noise_rate=args.noise_rate,
        noise_type=args.noise_type,
        score_based_noise=args.score_based
    )