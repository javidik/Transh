import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from context_management_layer import ContextManager

class RMSNorm(nn.Module):
    """RMSNorm normalization layer"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)

class GatedDeltaNet(nn.Module):
    """Gated Linear Recurrent Unit based on DeltaNet concept"""
    def __init__(self, d_model: int, d_state: int = 128, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * 2)  # For gate and value
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Convolution layer
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv-1, groups=d_model)
        
        # State projection
        self.state_proj = nn.Linear(d_model, d_state * 4)  # For computing A, B, dt parameters
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape  # Batch, Length, Dimension
        
        # Project input to gate and value
        gate_value = self.in_proj(x)  # [B, L, 2*D]
        gate, value = gate_value.chunk(2, dim=-1)  # Each [B, L, D]
        
        # Apply convolution (need to transpose for Conv1d)
        value_conv = value.transpose(1, 2)  # [B, D, L]
        value_conv = self.conv(value_conv)  # [B, D, L + d_conv - 1]
        value_conv = value_conv[:, :, :L]  # [B, D, L] - truncate back to original length
        value_conv = value_conv.transpose(1, 2)  # [B, L, D]
        
        # Apply gating
        gated_value = value_conv * F.silu(gate)  # [B, L, D]
        
        # Output projection
        output = self.out_proj(gated_value)
        
        return output

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) implementation"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_repeats = n_heads // n_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        queries = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        keys = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Repeat K and V heads to match Q heads
        if self.n_repeats > 1:
            keys = keys.repeat_interleave(self.n_repeats, dim=2)
            values = values.repeat_interleave(self.n_repeats, dim=2)
        
        # Transpose for attention calculation
        queries = queries.transpose(1, 2)  # [B, n_heads, L, head_dim]
        keys = keys.transpose(1, 2)        # [B, n_heads, L, head_dim]
        values = values.transpose(1, 2)    # [B, n_heads, L, head_dim]
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale  # [B, n_heads, L, L]
        
        if mask is not None:
            # Expand mask to match attention weights shape
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, values)  # [B, n_heads, L, head_dim]
        output = output.transpose(1, 2).contiguous()  # [B, L, n_heads, head_dim]
        output = output.view(batch_size, seq_len, self.d_model)  # [B, L, d_model]
        
        # Output projection
        output = self.o_proj(output)
        
        return output

class MixtureOfExperts(nn.Module):
    """Mixture of Experts Feed-Forward Network"""
    def __init__(self, d_model: int, n_experts: int, expert_used: int, ffn_multiplier: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.expert_used = expert_used
        self.ffn_hidden_dim = d_model * ffn_multiplier
        
        # Expert networks - using a parameter-efficient approach
        self.w_gate = nn.Parameter(torch.zeros(n_experts, d_model))  # Router weights
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, self.ffn_hidden_dim),
                nn.GELU(),
                nn.Linear(self.ffn_hidden_dim, d_model)
            ) for _ in range(n_experts)
        ])
        
        self.init_weights()
    
    def init_weights(self):
        # Initialize gate weights
        nn.init.xavier_uniform_(self.w_gate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for routing
        flat_x = x.view(-1, d_model)  # [B*L, d_model]
        
        # Compute gate scores for each expert
        gate_scores = torch.matmul(flat_x, self.w_gate.t())  # [B*L, n_experts]
        gate_scores = F.softmax(gate_scores, dim=-1)
        
        # Select top-k experts for each token
        topk_scores, topk_indices = torch.topk(gate_scores, self.expert_used, dim=-1)  # [B*L, expert_used]
        
        # Create a sparse representation of which tokens go to which experts
        flat_output = torch.zeros_like(flat_x)
        
        # Process each expert separately
        for i in range(self.expert_used):
            # Get the expert index for this iteration
            expert_idx = topk_indices[:, i]  # [B*L]
            
            # Create a mask for tokens assigned to this expert
            for exp_num in range(self.n_experts):
                mask = (expert_idx == exp_num)
                if mask.any():
                    # Get tokens assigned to this expert
                    expert_input = flat_x[mask]  # [num_assigned, d_model]
                    
                    # Process through the expert
                    expert_output = self.experts[exp_num](expert_input)  # [num_assigned, d_model]
                    
                    # Apply gate score
                    gate_val = topk_scores[mask, i].unsqueeze(-1)  # [num_assigned, 1]
                    expert_output = expert_output * gate_val
                    
                    # Add to the output
                    flat_output[mask] += expert_output
        
        # Reshape back to original shape
        output = flat_output.view(batch_size, seq_len, d_model)
        
        return output

class MultiTokenPredictionHead(nn.Module):
    """Multi-Token Prediction Head for next-token prediction"""
    def __init__(self, d_model: int, n_predict: int = 8):
        super().__init__()
        self.n_predict = n_predict
        
        # Multiple linear heads for predicting different numbers of tokens ahead
        self.prediction_heads = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_predict)
        ])
        
        # Final projection to vocabulary space would be handled by unembedding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # For each position, compute predictions for multiple future tokens
        predictions = []
        for i, head in enumerate(self.prediction_heads):
            # Shift input to predict i+1 tokens ahead
            if seq_len > i:
                shifted_input = x[:, :-i-1, :] if i > 0 else x  # Remove last i+1 elements
                pred = head(shifted_input)
                predictions.append(pred)
        
        # Return the first prediction head's output (can be extended to combine multiple)
        if predictions:
            return predictions[0]
        else:
            return x

class TransformerBlock(nn.Module):
    """A single transformer block with integrated context management"""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, n_experts: int, 
                 expert_used: int, ffn_multiplier: int, d_conv: int, d_state: int, dropout: float = 0.1):
        super().__init__()
        
        # Sublayer components
        self.rms_norm1 = RMSNorm(d_model)
        self.rms_norm2 = RMSNorm(d_model)
        self.rms_norm3 = RMSNorm(d_model)
        self.rms_norm4 = RMSNorm(d_model)
        
        # Attention mechanism
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        
        # Gated DeltaNet
        self.deltanet = GatedDeltaNet(d_model, d_state, d_conv)
        
        # Mixture of Experts Feed-Forward
        self.moe_ffn = MixtureOfExperts(d_model, n_experts, expert_used, ffn_multiplier)
        
        # Multi-token prediction head
        self.multi_token_pred = MultiTokenPredictionHead(d_model)
        
        # New Eisenhower Context Management Layer
        self.context_manager = ContextManager(d_model, n_heads=n_heads//2)  # Using fewer heads for efficiency
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        # Pre-normalization and attention
        residual1 = x
        x = self.rms_norm1(x)
        attn_out = self.attention(x, mask)
        x = residual1 + self.dropout(attn_out)
        
        # Context management
        residual2 = x
        x, context_analysis = self.context_manager(x, mask)
        x = residual2 + self.dropout(x)
        
        # Gated DeltaNet
        residual3 = x
        x = self.rms_norm2(x)
        deltanet_out = self.deltanet(x)
        x = residual3 + self.dropout(deltanet_out)
        
        # Mixture of Experts FFN
        residual4 = x
        x = self.rms_norm3(x)
        moe_out = self.moe_ffn(x)
        x = residual4 + self.dropout(moe_out)
        
        # Multi-token prediction (for training/inference)
        x = self.rms_norm4(x)
        pred_out = self.multi_token_pred(x)
        
        return x, context_analysis

class TransformerModel(nn.Module):
    """Complete Transformer Model with Eisenhower Context Management"""
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_blocks: int = 12,
                 n_kv_heads: int = 4,
                 n_experts: int = 8,
                 expert_used: int = 2,
                 ffn_multiplier: int = 4,
                 d_conv: int = 4,
                 d_state: int = 128,
                 seq_len: int = 4096,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Initial embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer blocks with integrated context management
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                n_experts=n_experts,
                expert_used=expert_used,
                ffn_multiplier=ffn_multiplier,
                d_conv=d_conv,
                d_state=d_state,
                dropout=dropout
            ) for _ in range(n_blocks)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(d_model)
        
        # Output layer (unembedding)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        # Initialize embedding weights
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        
        # Initialize positional encoding with sine/cosine (or random)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pos_enc = torch.zeros(1, self.seq_len, self.d_model)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 != 0:
            pos_enc[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        
        self.pos_encoding.data.copy_(pos_enc)
        
        # Tie embedding and output weights
        self.output.weight = self.embedding.weight
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        batch_size, seq_len = input_ids.shape
        
        # Embedding and positional encoding
        x = self.embedding(input_ids)  # [B, L, d_model]
        
        # Add positional encoding (truncate if needed)
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # Process through all blocks
        all_context_analyses = []
        for block in self.blocks:
            x, context_analysis = block(x, attention_mask)
            all_context_analyses.append(context_analysis)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output(x)  # [B, L, vocab_size]
        
        return logits, all_context_analyses

# Example usage and testing
if __name__ == "__main__":
    # Test the model
    model = TransformerModel(
        vocab_size=100000,
        d_model=1024,
        n_heads=16,
        n_blocks=4,  # Using fewer blocks for testing
        n_kv_heads=4,
        n_experts=8,
        expert_used=2,
        ffn_multiplier=4,
        d_conv=4,
        d_state=128,
        seq_len=2048
    )
    
    # Generate dummy input
    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, 100000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    
    # Forward pass
    logits, context_analyses = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of context analyses: {len(context_analyses)}")
    if context_analyses:
        print(f"Sample context analysis keys: {list(context_analyses[0].keys())}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print("Transformer model with integrated Eisenhower Context Management created successfully!")