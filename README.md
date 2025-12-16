# KGER: Knowledge Graph Error Detection and Refinement with Reinforcement Learning

Deep Reinforcement Learning-based Knowledge Graph Error Detection and Correction 

## Project Overview

This project implements an advanced knowledge graph error correction system that combines TransE embedding models with Deep Q-Networks (DQN), supporting both labeled and unlabeled training modes. The system can automatically detect and correct erroneous triples in knowledge graphs.

## Core Features

### 1. Staged Training
- **Stage 1**: Fully supervised with labels (label_weight=1.0)
- **Stage 2**: Partially supervised (label_weight=0.5)
- **Stage 3**: Unsupervised (label_weight=0.0, using only TransE features)

### 2. Improved Training Methods
- **Validation-based Model Selection**: Uses validation set F1 score to select the best model
- **Early Stopping**: Prevents overfitting
- **Best Model Saving per Stage**: Allows comparison of different supervision levels

### 3. Comprehensive Experimental Framework
- Modular experiment design
- 6 independent experiment scripts corresponding to paper sections
- Automated result visualization and LaTeX table generation

## System Architecture

```
advance_version/
├── models/                   # Core models
│   ├── transe.py            # TransE embedding model
│   └── agent.py             # DQN agent (supports Dueling architecture)
├── environment/              # Reinforcement learning environment
│   ├── kg_env.py            # Base KG environment
│   └── labeled_kg_env.py    # Labeled KG environment
├── utils/                    # Utility classes
│   ├── data_loader.py       # Data loading
│   ├── labeled_kg_dataset.py # Labeled dataset
│   └── evaluation.py        # Evaluation tools
├── train_transe.py           # TransE training script
├── train_dqn.py              # Standard DQN training
├── train_dqn_staged.py       # Staged DQN training
├── train_dqn_staged_improved.py # Improved staged training (recommended)
├── run_all_experiments.py    # Run all experiments
└── config.py                 # Configuration file
```

## Installation

```bash
pip install torch numpy matplotlib tqdm scipy seaborn tabulate
```

## Quick Start

### 1. Train TransE Model

```bash
python train_transe.py
```

### 2. Train DQN Agent (Improved version recommended)

```bash
# Improved staged training (uses validation set for best model selection)
python train_dqn_staged_improved.py

# Standard training
python train_dqn.py
```



## Configuration Parameters

Main configurations in `config.py`:

### TransE Parameters
```python
embedding_dim = 128      # Embedding dimension
margin = 1.0            # Margin ranking loss boundary
learning_rate = 0.0005  # Learning rate
num_epochs = 300        # Training epochs
```

### DQN Parameters
```python
hidden_dims = [512, 256, 128]  # Network architecture
learning_rate = 0.0001          # Learning rate
epsilon_start = 1.0             # Initial exploration rate
epsilon_end = 0.01              # Final exploration rate
gamma = 0.99                    # Discount factor
```

### Staged Training Parameters
```python
stage1_episodes = 300           # Stage 1 training episodes
stage1_label_weight = 1.0       # Stage 1 label weight (fully supervised)
stage2_episodes = 300           # Stage 2 training episodes
stage2_label_weight = 0.5       # Stage 2 label weight (partially supervised)
stage3_episodes = 300           # Stage 3 training episodes
stage3_label_weight = 0.0       # Stage 3 label weight (unsupervised)
```



## Usage Recommendations

1. **First-time Use**: Test the workflow with a small dataset first
2. **TransE Pre-training**: Ensure sufficient training (300+ epochs)
3. **Validation Set Usage**: Use the improved training script for validation-based model selection
4. **Noise Rate Setting**: Adjust training noise rate based on actual data quality
5. **Early Stopping Configuration**: Adjust patience parameter based on dataset size



## License

MIT License

## Contact

For questions, please submit an Issue or contact the author.
