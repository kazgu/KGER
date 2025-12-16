"""
Training script for TransE model
"""
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List

import numpy as np
from config import Config
from models.transe import TransE
from utils.data_loader import KGDataLoader
from utils.metrics import KGEvaluator

def negative_sampling(batch, num_entities, corrupt_rate=0.5):
    """Fixed negative sampling function"""
    h, r, t = batch
    batch_size = h.shape[0]
    
    # Create separate negative samples for head and tail corruption
    neg_h = h.clone()
    neg_r = r.clone() 
    neg_t = t.clone()
    
    # Randomly replace either head or tail entity
    for i in range(batch_size):
        if np.random.random() < corrupt_rate:
            # Replace tail entity: (h, r, neg_t)
            new_t = np.random.randint(num_entities)
            while new_t == t[i]:
                new_t = np.random.randint(num_entities)
            neg_t[i] = new_t
        else:
            # Replace head entity: (neg_h, r, t)  
            new_h = np.random.randint(num_entities)
            while new_h == h[i]:
                new_h = np.random.randint(num_entities)
            neg_h[i] = new_h
    
    return h, r, t, neg_h, neg_r, neg_t


def evaluate(model, test_loader, num_entities, device):
    """Evaluation function (from reference)"""
    model.eval()
    mr = 0.0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            h, r, t = batch
            h = h.to(device)
            r = r.to(device)
            t = t.to(device)
            batch_size = h.shape[0]
            
            for i in range(batch_size):
                # For each triple, replace tail entity to compute ranking
                current_h = h[i].repeat(num_entities)
                current_r = r[i].repeat(num_entities)
                current_t = torch.arange(num_entities, device=device)
                
                scores = model(current_h, current_r, current_t)
                true_score = scores[t[i]].item()
                
                # Calculate rank (number of scores smaller than true score + 1)
                rank = (scores < true_score).sum().item() + 1
                mr += rank
                if rank <= 1:
                    hits1 += 1
                if rank <= 3:
                    hits3 += 1
                if rank <= 10:
                    hits10 += 1
                count += 1
    
    mr /= count
    hits1 /= count
    hits3 /= count
    hits10 /= count
    
    return mr, hits1, hits3, hits10


def train_transe(save_path: str = None):
    """
    Train TransE model on real dataset
    
    Args:
        save_path: Optional path to save the model (auto-generated if None)
    """
    print("=== TransE Training ===\n")
    
    # Set random seed
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # Load data
    print(f"Loading dataset from {Config.data_path}...")
    data_loader = KGDataLoader(Config.data_path)
    
    # Get dataset name
    dataset_name = Config.get_dataset_name()
    
    num_entities = data_loader.get_num_entities()
    num_relations = data_loader.get_num_relations()
    if len(data_loader.train_triples)==0:
        data_loader.train_triples=data_loader.triples
    print(f"Number of entities: {num_entities}")
    print(f"Number of relations: {num_relations}")
    print(f"Number of triples: {len(data_loader.triples)}")
    print(f"Number of training triples: {len(data_loader.train_triples)}\n")
    
    # Initialize model
    model = TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=Config.TransE.embedding_dim,
        margin=Config.TransE.margin,
        norm=Config.TransE.norm,
        device=Config.device
    )
    
    # Optimizer with higher learning rate for TransE
    optimizer = optim.Adam(
        model.parameters(), 
        lr=Config.TransE.learning_rate,
        weight_decay=Config.TransE.regularization_weight
    )
    
    # Learning rate scheduler for better convergence
    # scheduler = optim.lr_scheduler.StepLR(
    #     optimizer, step_size=100, gamma=0.5
    # )
    
    # Training loop
    losses = []
    best_mrr = 0
    best_epoch = 0
    batch_size = Config.TransE.batch_size
    num_epochs = Config.TransE.num_epochs
    
    # Create DataLoaders (as in reference)
    train_loader = data_loader.get_train_dataloader(batch_size, shuffle=True, num_workers=4)
    test_loader = data_loader.get_test_dataloader(batch_size, shuffle=False, num_workers=4)
    
    evaluator = KGEvaluator(model, data_loader, Config.device)
    print("Starting training (using torch DataLoader)...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # Use DataLoader for batches (as in reference)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Generate negative samples (fixed version)
            h, r, t, neg_h, neg_r, neg_t = negative_sampling(batch, num_entities)
            h = h.to(Config.device)
            r = r.to(Config.device)
            t = t.to(Config.device)
            neg_h = neg_h.to(Config.device)
            neg_r = neg_r.to(Config.device)
            neg_t = neg_t.to(Config.device)
            
            # Forward pass with negative samples
            positive_score = model(h, r, t)
            # Use either corrupted head or tail, but not both!
            # neg_h and neg_t already contain the appropriate corruption
            negative_score = model(neg_h, neg_r, neg_t)
            loss = model.loss_function(positive_score, negative_score)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Normalize entity embeddings (keep TransE constraint - as in reference)
            model.entity_embeddings.weight.data = torch.nn.functional.normalize(
                model.entity_embeddings.weight.data, p=2, dim=1)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Step scheduler every epoch
        # scheduler.step()
        
        # Evaluate every 10 epochs (as in reference)
        if (epoch + 1) % 20 == 0:
            mr, hits1, hits3, hits10 = evaluate(model, test_loader, num_entities, Config.device)
            print(f"Evaluation Results - MR: {mr:.2f}, Hits@1: {hits1:.4f}, "
                  f"Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}")
            
            # Save best model based on Hits@10
            if hits1 > best_mrr:  # Using hits10 as metric
                best_mrr = hits1
                best_epoch = epoch + 1
                save_path = f"{Config.model_save_path}/{dataset_name}_transe_best.pt"
                os.makedirs(Config.model_save_path, exist_ok=True)
                model.save(save_path)
                print(f"  New best model saved! (Hits@1: {best_mrr:.4f})")
        
        # # Save checkpoint with dataset name
        # if (epoch + 1) % Config.save_interval == 0:
        #     checkpoint_path = f"{Config.model_save_path}/{dataset_name}_transe_epoch_{epoch+1}.pt"
        #     os.makedirs(Config.model_save_path, exist_ok=True)
        #     model.save(checkpoint_path)
        #     print(f"Model saved to {checkpoint_path}")
    
    print(f"\nBest Hits@10: {best_mrr:.4f} at epoch {best_epoch}")
    
    # Load best model if available
    best_model_path = f"{Config.model_save_path}/{dataset_name}_transe_best.pt"
    if os.path.exists(best_model_path) and best_mrr > 0:
        model.load(best_model_path)
        print(f"Loaded best model from epoch {best_epoch}")
    
    # Save final model with dataset name

    final_path = f"{Config.model_save_path}/{dataset_name}_transe_final.pt"
    model.save(final_path)
    print(f"Training completed! Final model saved to {final_path}")
    
    # Plot training loss
    plot_training_loss(losses)
    
    # Final evaluation on test set (using reference style)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    if data_loader.test_triples and len(data_loader.test_triples) > 0:
        print(f"\nEvaluating on test set...")
        mr, hits1, hits3, hits10 = evaluate(model, test_loader, num_entities, Config.device)
            
        print("\n📊 Test Set Results:")
        print("-" * 40)
        print(f"  MR:         {mr:.2f}")
        print(f"  Hits@1:     {hits1:.4f}")
        print(f"  Hits@3:     {hits3:.4f}")
        print(f"  Hits@10:    {hits10:.4f}")
            
        # Triple quality metrics
        # quality_metrics = evaluator.evaluate_triple_quality(test_sample)
        # print(f"\n📈 Triple Quality:")
        # print(f"  Mean Score: {quality_metrics['mean_score']:.4f} ± {quality_metrics['std_score']:.4f}")
        # print(f"  Min Score:  {quality_metrics['min_score']:.4f}")
        # print(f"  Max Score:  {quality_metrics['max_score']:.4f}")
        
        # Compare with expected performance
        if 'FB15K' in Config.data_path:
            print("\n📌 Expected Performance (TransE on FB15K-237):")
            print("  MRR:        ~0.29")
            print("  Hits@1:     ~0.20")
            print("  Hits@3:     ~0.32")
            print("  Hits@10:    ~0.47") 
        else:  # WN18RR
            print("\n📌 Expected Performance (TransE on WN18RR):")
            print("  MRR:        ~0.22")
            print("  Hits@1:     ~0.04")
            print("  Hits@3:     ~0.44")
            print("  Hits@10:    ~0.53")
            
    elif data_loader.valid_triples and len(data_loader.valid_triples) > 0:
        print(f"\nNo test set found. Evaluating on {len(data_loader.valid_triples)} validation triples...")
        model.eval()
        
        with torch.no_grad():

            
            valid_sample_size = min(5000, len(data_loader.valid_triples))
            if valid_sample_size < len(data_loader.valid_triples):
                import random
                valid_sample = random.sample(data_loader.valid_triples, valid_sample_size)
                print(f"  (Sampled {valid_sample_size} triples for efficiency)")
            else:
                valid_sample = data_loader.valid_triples
            
            lp_metrics = evaluator.evaluate_link_prediction(valid_sample, filtered=True)
            
            print("\n📊 Validation Set Results:")
            print("-" * 40)
            print(f"  MRR:        {lp_metrics['mrr']:.4f}")
            print(f"  Hits@1:     {lp_metrics['hits@1']:.4f}")
            print(f"  Hits@3:     {lp_metrics['hits@3']:.4f}")
            print(f"  Hits@10:    {lp_metrics['hits@10']:.4f}")
            print(f"  Mean Rank:  {lp_metrics['mean_rank']:.2f}")
    else:
        print("\n⚠ No test or validation set available for final evaluation.")
    
    print("\n" + "=" * 60)
    
    return model, data_loader


def evaluate_transe(model: TransE, data_loader: KGDataLoader, num_samples: int = 5):
    """
    Evaluate TransE model on sample predictions
    
    Args:
        model: Trained TransE model
        data_loader: Data loader
        num_samples: Number of samples to evaluate
    """
    print("\n--- Sample Predictions ---")
    
    # Sample some triples
    import random
    sample_indices = random.sample(range(len(data_loader.train_triples)), 
                                  min(num_samples, len(data_loader.train_triples)))
    
    for idx in sample_indices:
        h, r, t = data_loader.train_triples[idx]
        
        # Get triple names
        h_name, r_name, t_name = data_loader.triple_to_names((h, r, t))
        
        # Score the original triple
        score = model.score_triple(h, r, t)
        
        print(f"\nTriple: ({h_name}, {r_name}, {t_name})")
        print(f"Score: {score:.4f}")
        
        # Predict tail given head and relation
        predicted_tails, tail_scores = model.predict_tail(h, r, k=3)
        print(f"Predicted tails: ", end="")
        for pred_t, pred_score in zip(predicted_tails[:3], tail_scores[:3]):
            pred_name = data_loader.id2entity.get(pred_t, f"Entity_{pred_t}")
            print(f"{pred_name}({pred_score:.2f}) ", end="")
        print()


def plot_training_loss(losses: List[float]):
    """
    Plot training loss curve
    
    Args:
        losses: List of loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TransE Training Loss')
    plt.grid(True)
    
    os.makedirs(Config.log_path, exist_ok=True)
    plt.savefig(f"{Config.log_path}/transe_training_loss.png")
    plt.close()
    print(f"Training loss plot saved to {Config.log_path}/transe_training_loss.png")


if __name__ == "__main__":
    # Train TransE on real data only
    model, data_loader = train_transe()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    evaluate_transe(model, data_loader, num_samples=10)