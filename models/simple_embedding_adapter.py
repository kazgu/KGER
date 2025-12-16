"""
Simple Embedding Adapter - 最直接最简单的维度转换
在KG embedding获取后、进入DQN前进行统一维度转换
"""
import torch
import torch.nn as nn


class SimpleEmbeddingAdapter(nn.Module):
    """
    简单的embedding维度转换adapter
    - 输入维度自适应（自动检测KG embedding维度）
    - 输出维度固定（可配置，默认128）
    - 无需判断模型类型，统一处理所有embedding
    """
    
    def __init__(self, input_dim: int, output_dim: int = 128, dropout: float = 0.1):
        """
        初始化简单adapter
        
        Args:
            input_dim: 输入embedding维度（自动检测得到）
            output_dim: 输出维度（标准化维度）
            dropout: Dropout率
        """
        super(SimpleEmbeddingAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 简单的MLP: Linear -> ReLU -> Dropout
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 将任意维度的embedding转换为标准维度
        
        Args:
            embedding: 输入embedding tensor [embedding_dim] 或 [batch_size, embedding_dim]
            
        Returns:
            转换后的embedding tensor [output_dim] 或 [batch_size, output_dim]
        """
        # 确保embedding是扁平的
        if embedding.dim() > 1:
            original_shape = embedding.shape[:-1]  # 保存批次维度
            embedding = embedding.view(-1, self.input_dim)
            output = self.adapter(embedding)
            return output.view(*original_shape, self.output_dim)
        else:
            return self.adapter(embedding.view(1, -1)).squeeze(0)
    
    def extra_repr(self) -> str:
        """额外的表示信息"""
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}'


def create_embedding_adapter(kg_model, output_dim: int = 128) -> SimpleEmbeddingAdapter:
    """
    自动创建embedding adapter
    
    Args:
        kg_model: KG模型（用于自动检测embedding维度）
        output_dim: 输出维度
        
    Returns:
        配置好的SimpleEmbeddingAdapter
    """
    # 自动检测embedding维度
    with torch.no_grad():
        # 使用第一个实体/关系来检测维度
        h_emb, r_emb, t_emb = kg_model.get_embeddings(0, 0, 1)
        
        # 获取扁平化后的维度
        embedding_dim = h_emb.flatten().shape[0]
        
        print(f"   Auto-detected embedding dimension: {embedding_dim}")
        print(f"   Creating adapter: {embedding_dim} -> {output_dim}")
    
    return SimpleEmbeddingAdapter(
        input_dim=embedding_dim,
        output_dim=output_dim,
        dropout=0.1
    )