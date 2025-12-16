"""
Main entry point for the Advanced KG Error Correction System
"""
import os
import sys
import argparse
import torch
import numpy as np

from config import Config
from train_transe import train_transe
from train_dqn import train_dqn
from evaluate import evaluate_system


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced KG Error Correction System')
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'transe', 'dqn', 'eval'],
                       help='Execution mode: full (train everything), transe (train TransE only), '
                            'dqn (train DQN only), eval (evaluate only)')
    
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to dataset directory')
    
    parser.add_argument('--transe_epochs', type=int, default=None,
                       help='Number of epochs for TransE training')
    
    parser.add_argument('--dqn_episodes', type=int, default=None,
                       help='Number of episodes for DQN training')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Update configuration
    if args.data_path:
        Config.data_path = args.data_path
    
    if args.transe_epochs:
        Config.TransE.num_epochs = args.transe_epochs
    
    if args.dqn_episodes:
        Config.DQN.num_episodes = args.dqn_episodes
    
    if args.device:
        Config.device = torch.device(args.device)
    
    Config.seed = args.seed
    
    # Set random seeds
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    print("=" * 60)
    print("Advanced KG Error Correction System")
    print("Based on DQN + TransE")
    print("=" * 60)
    print()
    
    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {Config.device}")
    print(f"  Data path: {Config.data_path}")
    print(f"  Random seed: {Config.seed}")
    print()
    
    # Create directories
    os.makedirs(Config.model_save_path, exist_ok=True)
    os.makedirs(Config.log_path, exist_ok=True)
    
    # Execute based on mode
    if args.mode == 'full':
        print("=== Full Training Pipeline ===\n")
        
        # Step 1: Train TransE
        print("Step 1: Training TransE model...")
        print("-" * 40)
        transe_model, data_loader = train_transe()
        print("\n" + "=" * 60 + "\n")
        
        # Step 2: Train DQN
        print("Step 2: Training DQN agent...")
        print("-" * 40)
        transe_path = f"{Config.model_save_path}/transe_final.pt"
        agent, env = train_dqn(transe_path=transe_path)
        print("\n" + "=" * 60 + "\n")
        
        # Step 3: Evaluate
        print("Step 3: System Evaluation...")
        print("-" * 40)
        dqn_path = f"{Config.model_save_path}/dqn_final.pt"
        evaluate_system(transe_path, dqn_path)
        
    elif args.mode == 'transe':
        print("=== TransE Training Only ===\n")
        train_transe()
        
    elif args.mode == 'dqn':
        print("=== DQN Training Only ===\n")
        transe_path = f"{Config.model_save_path}/transe_final.pt"
        
        if not os.path.exists(transe_path):
            print("Error: TransE model not found. Please train TransE first.")
            print("Run: python main.py --mode transe")
            return
        
        train_dqn(transe_path=transe_path)
        
    elif args.mode == 'eval':
        print("=== Evaluation Only ===\n")
        transe_path = f"{Config.model_save_path}/transe_final.pt"
        dqn_path = f"{Config.model_save_path}/dqn_final.pt"
        
        evaluate_system(transe_path, dqn_path)
    
    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()