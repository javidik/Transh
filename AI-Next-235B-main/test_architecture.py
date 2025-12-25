"""
Simplified test version of the world's best neural network to demonstrate the architecture.
"""

import math
import numpy as np
from typing import Optional, Tuple, List


class RMSNorm:
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim)  # Initialize learnable weight parameter
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Calculate RMS
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)


class GatedDeltaNet:
    """
    Gated Linear Attention mechanism
    """
    def __init__(self, dim: int, head_dim: int = 16, num_heads: int = 4):
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        # Initialize weights for Q, K, V projections
        self.w_q = np.random.randn(dim, head_dim * num_heads) * 0.02
        self.w_k = np.random.randn(dim, head_dim * num_heads) * 0.02
        self.w_v = np.random.randn(dim, head_dim * num_heads) * 0.02
        self.w_gate = np.random.randn(dim, head_dim * num_heads) * 0.02
        
        # Output projection
        self.w_out = np.random.randn(head_dim * num_heads, dim) * 0.02
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, C = x.shape
        
        # Project to Q, K, V and gate
        q = (x @ self.w_q).reshape(B, T, self.num_heads, self.head_dim)  # [B, T, H, D]
        k = (x @ self.w_k).reshape(B, T, self.num_heads, self.head_dim)  # [B, T, H, D]
        v = (x @ self.w_v).reshape(B, T, self.num_heads, self.head_dim)  # [B, T, H, D]
        
        # Compute gate
        gate_input = (x @ self.w_gate).reshape(B, T, self.num_heads, self.head_dim)  # [B, T, H, D]
        
        # Apply linear attention mechanism
        # Using exponential moving average for efficiency
        q = np.where(q > 0, q, np.exp(q) - 1) + 1.0  # ELU + 1
        k = np.where(k > 0, k, np.exp(k) - 1) + 1.0  # ELU + 1
        
        # Compute attention in linear time
        k_cumsum = np.cumsum(k, axis=1)  # [B, T, H, D]
        kv = np.cumsum(k[:, :, :, np.newaxis, :] * v[:, :, :, :, np.newaxis], axis=1)  # [B, T, H, D, D]
        
        # Compute numerator and denominator
        numerator = np.sum(q[:, :, :, :, np.newaxis] * kv, axis=-1)  # [B, T, H, D]
        denominator = np.sum(q[:, :, :, :, np.newaxis] * k_cumsum[:, :, :, :, np.newaxis], axis=-1)  # [B, T, H, D]
        
        # Compute output
        output = numerator / (denominator + 1e-6)
        
        # Apply gating mechanism
        gate = 1 / (1 + np.exp(-gate_input))  # Sigmoid
        output = output * gate
        
        # Reshape and project back
        output = output.reshape(B, T, self.num_heads * self.head_dim)
        return output @ self.w_out


class MixtureOfExperts:
    """
    High-capacity Mixture of Experts layer
    """
    def __init__(self, dim: int, num_experts: int = 4, expert_size: int = 64, top_k: int = 2):
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.top_k = top_k
        
        # Initialize experts
        self.experts = []
        for _ in range(num_experts):
            w1 = np.random.randn(dim, expert_size) * 0.02
            b1 = np.zeros(expert_size)
            w2 = np.random.randn(expert_size, dim) * 0.02
            b2 = np.zeros(dim)
            self.experts.append((w1, b1, w2, b2))
        
        # Router network to compute expert weights
        self.router_w = np.random.randn(dim, num_experts) * 0.02
        self.router_b = np.zeros(num_experts)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, C = x.shape
        
        # Compute routing weights
        router_logits = (x @ self.router_w) + self.router_b  # [B, T, num_experts]
        
        # Get top-k experts for each token
        top_indices = np.argpartition(router_logits, -self.top_k, axis=-1)[:, :, -self.top_k:]  # [B, T, k]
        
        # Get values for top-k
        top_values = np.take_along_axis(router_logits, top_indices, axis=-1)  # [B, T, k]
        
        # Apply softmax to get weights
        top_values_exp = np.exp(top_values - np.max(top_values, axis=-1, keepdims=True))
        top_weights = top_values_exp / np.sum(top_values_exp, axis=-1, keepdims=True)  # [B, T, k]
        
        # Process tokens through selected experts
        final_output = np.zeros_like(x)
        
        for i in range(self.top_k):
            # Get current expert indices and weights
            expert_idx = top_indices[:, :, i]  # [B, T]
            weight = top_weights[:, :, i, np.newaxis]  # [B, T, 1]
            
            # Process each token through its selected expert
            for b in range(B):
                for t in range(T):
                    expert_id = expert_idx[b, t]
                    expert_input = x[b, t, :]  # [C]
                    
                    # Get expert parameters
                    w1, b1, w2, b2 = self.experts[expert_id]
                    
                    # Forward through expert
                    h = expert_input @ w1 + b1
                    h = np.where(h > 0, h, 0.01 * h)  # Leaky ReLU equivalent
                    expert_output = h @ w2 + b2
                    
                    final_output[b, t, :] += weight[b, t, 0] * expert_output
        
        return final_output


class GroupedQueryAttention:
    """
    Grouped Query Attention
    """
    def __init__(self, dim: int, num_heads: int = 4, group_size: int = 2):
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = dim // num_heads
        
        # Number of key-value groups
        self.num_groups = max(1, num_heads // group_size)
        
        # Initialize projections
        self.w_q = np.random.randn(dim, num_heads * self.head_dim) * 0.02
        self.w_k = np.random.randn(dim, self.num_groups * self.head_dim) * 0.02
        self.w_v = np.random.randn(dim, self.num_groups * self.head_dim) * 0.02
        self.w_out = np.random.randn(num_heads * self.head_dim, dim) * 0.02
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = (x @ self.w_q).reshape(B, T, self.num_heads, self.head_dim)  # [B, T, H, D]
        k = (x @ self.w_k).reshape(B, T, self.num_groups, self.head_dim)  # [B, T, G, D]
        v = (x @ self.w_v).reshape(B, T, self.num_groups, self.head_dim)  # [B, T, G, D]
        
        # Expand K and V to match number of heads
        k = np.repeat(k, self.num_heads // self.num_groups, axis=2)  # [B, T, H, D]
        v = np.repeat(v, self.num_heads // self.num_groups, axis=2)  # [B, T, H, D]
        
        # Transpose for attention computation: [B, H, T, D]
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        
        # Compute attention scores
        attn_scores = (q @ np.transpose(k, (0, 1, 3, 2)))  # [B, H, T, T]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = np.where(mask == 0, -1e9, attn_scores)
        
        # Apply softmax
        attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # Apply attention to values
        output = attn_weights @ v  # [B, H, T, D]
        
        # Reshape and project back
        output = np.transpose(output, (0, 2, 1, 3)).reshape(B, T, -1)  # [B, T, H*D]
        return output @ self.w_out


class MultiTokenPredictionHead:
    """
    Specialized head that allows predicting multiple tokens simultaneously
    """
    def __init__(self, dim: int, prediction_horizon: int = 2, vocab_size: int = 1000):
        self.prediction_horizon = prediction_horizon
        self.vocab_size = vocab_size
        
        # Initialize predictors for multiple future tokens
        self.token_predictors = []
        for _ in range(prediction_horizon):
            w1 = np.random.randn(dim, dim) * 0.02
            b1 = np.zeros(dim)
            w2 = np.random.randn(dim, vocab_size) * 0.02
            b2 = np.zeros(vocab_size)
            self.token_predictors.append((w1, b1, w2, b2))
        
    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        predictions = []
        for w1, b1, w2, b2 in self.token_predictors:
            h = x @ w1 + b1
            h = np.where(h > 0, h, 0.01 * h)  # Leaky ReLU equivalent
            pred = h @ w2 + b2
            predictions.append(pred)
        return predictions


class SuperBlock:
    """
    A single superblock containing 8 specialized layers as per your architecture
    """
    def __init__(self, dim: int, num_heads: int = 4, head_dim: int = 16, 
                 num_experts: int = 2, expert_size: int = 32, vocab_size: int = 1000):
        
        # Layer 1: RMSNorm
        self.norm1 = RMSNorm(dim)
        
        # Layer 2: Gated DeltaNet (Linear Attention)
        self.deltanet = GatedDeltaNet(dim, head_dim, num_heads)
        
        # Layer 3: MoE FFN Layer (High-Capacity)
        self.moe_ffn1 = MixtureOfExperts(dim, num_experts, expert_size)
        
        # Layer 4: RMSNorm
        self.norm2 = RMSNorm(dim)
        
        # Layer 5: Grouped Query Attention (GQA)
        self.gqa_attention = GroupedQueryAttention(dim, num_heads, group_size=2)
        
        # Layer 6: MoE FFN Layer (High-Capacity)
        self.moe_ffn2 = MixtureOfExperts(dim, num_experts, expert_size)
        
        # Layer 7: RMSNorm
        self.norm3 = RMSNorm(dim)
        
        # Layer 8: Multi-Token Prediction (MTP) Head
        self.mtp_head = MultiTokenPredictionHead(dim, prediction_horizon=2, vocab_size=vocab_size)
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        # Layer 1: RMSNorm
        residual1 = x.copy()
        x = self.norm1.forward(x)
        
        # Layer 2: Gated DeltaNet
        x = self.deltanet.forward(x) + residual1
        
        # Layer 3: MoE FFN
        residual2 = x.copy()
        x = self.moe_ffn1.forward(x) + residual2
        
        # Layer 4: RMSNorm
        residual3 = x.copy()
        x = self.norm2.forward(x)
        
        # Layer 5: GQA Attention
        x = self.gqa_attention.forward(x, mask) + residual3
        
        # Layer 6: MoE FFN
        residual4 = x.copy()
        x = self.moe_ffn2.forward(x) + residual4
        
        # Layer 7: RMSNorm
        x = self.norm3.forward(x)
        
        # Layer 8: MTP Head (returns predictions but keeps x for next block)
        predictions = self.mtp_head.forward(x)
        
        return x, predictions


class WorldClassNeuralNetwork:
    """
    The complete world-class neural network with the architecture you described:
    - 2 initial layers: Embedding and Positional Encoding
    - 2 identical superblocks (each with 8 layers = 16 total block layers) for testing
    - 3 final layers: Final RMSNorm, Output Layer (Unembedding), and Softmax/Sampling
    Total: ~21 layers in this test version
    """
    def __init__(self, 
                 vocab_size: int = 1000,  # Smaller vocab for testing
                 dim: int = 128,  # Much smaller dimension for testing
                 max_seq_len: int = 64,  # Shorter sequence for testing
                 num_blocks: int = 2,  # Only 2 blocks for testing
                 num_heads: int = 4,
                 head_dim: int = 16,
                 num_experts: int = 2,
                 expert_size: int = 32):
        
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        
        print(f"Initializing model with:")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Hidden dimension: {dim}")
        print(f"  - Max sequence length: {max_seq_len}")
        print(f"  - Number of blocks: {num_blocks}")
        print(f"  - Num heads: {num_heads}")
        print(f"  - Num experts: {num_experts}")
        
        # Initial layers (2 total)
        # Embedding Layer
        self.embedding = np.random.randn(vocab_size, dim) * 0.02
        
        # Positional Encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, dim)
        
        # Stack of superblocks (16 layers total in this test version)
        self.blocks = [
            SuperBlock(dim, num_heads, head_dim, num_experts, expert_size, vocab_size) 
            for _ in range(num_blocks)
        ]
        
        # Final layers (3 total)
        # Final RMSNorm
        self.final_norm = RMSNorm(dim)
        
        # Output Layer (Unembedding)
        self.unembedding = np.random.randn(dim, vocab_size) * 0.02
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encodings"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1).astype(float)
        
        div_term = np.exp(np.arange(0, d_model, 2).astype(float) * 
                         -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, 
                input_ids: np.ndarray, 
                attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        """
        Forward pass of the neural network
        """
        B, T = input_ids.shape
        
        print(f"Processing input with shape: {input_ids.shape}")
        
        # Initial processing: Embedding + Positional Encoding (2 layers)
        x = self.embedding[input_ids]  # [B, T, dim]
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:T, :]
        x = x + pos_enc
        
        # Process through all superblocks (2 blocks × 8 layers each = 16 layers)
        all_predictions = []
        for i, block in enumerate(self.blocks):
            print(f"Processing block {i+1}/{self.num_blocks}")
            x, predictions = block.forward(x, attention_mask)
            all_predictions.append(predictions)
        
        # Final processing (3 layers)
        # Final RMSNorm
        x = self.final_norm.forward(x)
        
        # Output Layer (Unembedding)
        logits = x @ self.unembedding  # [B, T, vocab_size]
        
        return logits, all_predictions


def count_parameters(model: WorldClassNeuralNetwork) -> int:
    """Count the total number of parameters in the model"""
    total_params = 0
    
    # Embedding layer
    total_params += model.embedding.size
    print(f"Embedding params: {model.embedding.size:,}")
    
    # Parameters in each superblock
    for i, block in enumerate(model.blocks):
        block_params = 0
        
        # norm1
        block_params += block.norm1.weight.size
        
        # deltanet
        block_params += (block.deltanet.w_q.size + block.deltanet.w_k.size + 
                        block.deltanet.w_v.size + block.deltanet.w_gate.size + 
                        block.deltanet.w_out.size)
        
        # moe_ffn1
        for w1, b1, w2, b2 in block.moe_ffn1.experts:
            block_params += w1.size + b1.size + w2.size + b2.size
        block_params += block.moe_ffn1.router_w.size + block.moe_ffn1.router_b.size
        
        # norm2
        block_params += block.norm2.weight.size
        
        # gqa_attention
        block_params += (block.gqa_attention.w_q.size + block.gqa_attention.w_k.size + 
                        block.gqa_attention.w_v.size + block.gqa_attention.w_out.size)
        
        # moe_ffn2
        for w1, b1, w2, b2 in block.moe_ffn2.experts:
            block_params += w1.size + b1.size + w2.size + b2.size
        block_params += block.moe_ffn2.router_w.size + block.moe_ffn2.router_b.size
        
        # norm3
        block_params += block.norm3.weight.size
        
        # mtp_head
        for w1, b1, w2, b2 in block.mtp_head.token_predictors:
            block_params += w1.size + b1.size + w2.size + b2.size
            
        total_params += block_params
        print(f"Block {i+1} params: {block_params:,}")
    
    # Final norm
    total_params += model.final_norm.weight.size
    print(f"Final norm params: {model.final_norm.weight.size:,}")
    
    # Unembedding layer
    total_params += model.unembedding.size
    print(f"Unembedding params: {model.unembedding.size:,}")
    
    return total_params


# Example usage and testing
if __name__ == "__main__":
    print("Creating the world's best neural network (test version)...")
    
    # Create a very small model for demonstration purposes
    model = WorldClassNeuralNetwork(
        vocab_size=1000,    # Small vocab
        dim=128,            # Small hidden dimension
        max_seq_len=64,     # Short sequence
        num_blocks=2,       # Just 2 blocks for testing
        num_heads=4,        # Fewer heads
        head_dim=16,        # Smaller head dimension
        num_experts=2,      # Fewer experts
        expert_size=32      # Smaller expert size
    )
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass with tiny inputs
    batch_size = 2
    seq_len = 8
    input_ids = np.random.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nTesting forward pass...")
    logits, predictions = model.forward(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of prediction sets: {len(predictions)}")  # Should be 2 blocks
    print(f"Predictions per block: {len(predictions[0])}")  # Should be 2 (prediction horizon)
    
    print("\nArchitecture Summary:")
    print("- 2 initial layers: Embedding and Positional Encoding")
    print("- 2 superblocks with 8 layers each (16 total block layers)")
    print("- 3 final layers: Final RMSNorm, Output Layer, Softmax")
    print("- Total layers: 21 (scalable to 101 with full 12-block configuration)")
    print("\nEach superblock contains:")
    print("  1. RMSNorm")
    print("  2. Gated DeltaNet (Linear Attention)")  
    print("  3. MoE FFN Layer (High-Capacity)")
    print("  4. RMSNorm")
    print("  5. Grouped Query Attention (GQA)")
    print("  6. MoE FFN Layer (High-Capacity)")
    print("  7. RMSNorm") 
    print("  8. Multi-Token Prediction (MTP) Head")
    print("\nThis demonstrates the complete architecture that can be scaled up to 235B parameters.")