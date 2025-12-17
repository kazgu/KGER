
import torch
import torch.nn as nn


class SimpleEmbeddingAdapter(nn.Module):

    
    def __init__(self, input_dim: int, output_dim: int = 128, dropout: float = 0.1):

        super(SimpleEmbeddingAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:


        if embedding.dim() > 1:
            original_shape = embedding.shape[:-1]  
            embedding = embedding.view(-1, self.input_dim)
            output = self.adapter(embedding)
            return output.view(*original_shape, self.output_dim)
        else:
            return self.adapter(embedding.view(1, -1)).squeeze(0)
    
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}'


def create_embedding_adapter(kg_model, output_dim: int = 128) -> SimpleEmbeddingAdapter:


    with torch.no_grad():

        h_emb, r_emb, t_emb = kg_model.get_embeddings(0, 0, 1)
        
 
        embedding_dim = h_emb.flatten().shape[0]
        
        print(f"   Auto-detected embedding dimension: {embedding_dim}")
        print(f"   Creating adapter: {embedding_dim} -> {output_dim}")
    
    return SimpleEmbeddingAdapter(
        input_dim=embedding_dim,
        output_dim=output_dim,
        dropout=0.1
    )