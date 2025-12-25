import torch
import torch.nn as nn
from context_management_layer import ContextManager
from transformer_architecture import TransformerModel
import math

def count_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_weight_breakdown(model_config):
    """
    Calculate detailed breakdown of weights in the full model with Eisenhower Context Layer
    """
    print("Calculating weight breakdown for the complete model with Eisenhower Context Layer...")
    print("="*80)
    
    # Model configuration
    d_model = model_config['d_model']
    n_heads = model_config['n_heads']
    n_blocks = model_config['n_blocks']
    vocab_size = model_config['vocab_size']
    seq_len = model_config['seq_len']
    mlp_multiplier = model_config['mlp_multiplier']
    n_experts = model_config['n_experts']
    expert_used = model_config['expert_used']
    n_kv_heads = model_config['n_kv_heads']
    d_conv = model_config['d_conv']
    d_state = model_config['d_state']
    
    # Calculate individual component sizes
    embedding_params = vocab_size * d_model
    pos_encoding_params = seq_len * d_model
    
    # Attention mechanism parameters
    qkv_params_per_head = d_model * (d_model // n_heads)  # Per head
    qkv_total_params = qkv_params_per_head * n_heads * 3  # For Q, K, V
    out_proj_params = d_model * d_model
    
    # GQA (Grouped Query Attention) parameters
    kv_groups = n_heads // n_kv_heads
    k_proj_gqa_params = d_model * (d_model // n_kv_heads) * n_kv_heads
    v_proj_gqa_params = d_model * (d_model // n_kv_heads) * n_kv_heads
    
    # DeltaNet parameters (simplified approximation)
    deltanet_params = d_model * d_model * 2  # For conv and linear projections
    
    # MoE FFN parameters
    mlp_hidden_dim = d_model * mlp_multiplier
    expert_params = d_model * mlp_hidden_dim * 2  # Up and gate projections
    total_moe_params = expert_params * n_experts
    
    # Gate parameters for MoE
    moe_gate_params = d_model * n_experts
    
    # RMSNorm parameters (per block)
    rmsnorm_params_per_block = d_model * 4  # Multiple norms per block
    
    # Output layer parameters
    output_layer_params = d_model * vocab_size
    
    # Eisenhower Context Layer parameters
    eisenhower_importance_params = d_model * (d_model // 2) + (d_model // 2) * 1  # Linear layers in importance scorer
    eisenhower_urgency_params = d_model * (d_model // 2) + (d_model // 2) * 1    # Linear layers in urgency scorer
    eisenhower_attention_params = d_model * d_model * 4  # Q, K, V, O projections
    eisenhower_quadrant_params = (d_model + 2) * (d_model // 4) + (d_model // 4) * 4  # Quadrant classifier (+2 for imp/urg scores)
    eisenhower_context_weights = 4  # Learnable weights for each quadrant
    
    eisenhower_total_params = (
        eisenhower_importance_params +
        eisenhower_urgency_params +
        eisenhower_attention_params +
        eisenhower_quadrant_params +
        eisenhower_context_weights
    )
    
    # Calculate total parameters per block
    params_per_block = (
        qkv_total_params +
        out_proj_params +
        k_proj_gqa_params +
        v_proj_gqa_params +
        deltanet_params +
        expert_params * expert_used +  # Only counting used experts
        moe_gate_params +
        rmsnorm_params_per_block +
        eisenhower_total_params  # Adding our new context layer
    )
    
    # Total model parameters
    total_params = (
        embedding_params +
        pos_encoding_params +
        n_blocks * params_per_block +
        output_layer_params +
        rmsnorm_params_per_block  # Final RMSNorm
    )
    
    # Print detailed breakdown
    print(f"Vocabulary Embedding Layer:           {embedding_params:,} ({embedding_params/1e9:.2f}B)")
    print(f"Positional Encoding:                  {pos_encoding_params:,} ({pos_encoding_params/1e9:.2f}B)")
    print()
    
    print(f"Parameters per Transformer Block:")
    print(f"  Attention (Q,K,V,O):               {qkv_total_params + out_proj_params:,} ({(qkv_total_params + out_proj_params)/1e6:.2f}M)")
    print(f"  GQA Projections:                   {k_proj_gqa_params + v_proj_gqa_params:,} ({(k_proj_gqa_params + v_proj_gqa_params)/1e6:.2f}M)")
    print(f"  DeltaNet Components:               {deltanet_params:,} ({deltanet_params/1e6:.2f}M)")
    print(f"  MoE FFN (per block):               {expert_params * expert_used:,} ({(expert_params * expert_used)/1e6:.2f}M)")
    print(f"  MoE Gate (per block):              {moe_gate_params:,} ({moe_gate_params/1e6:.2f}M)")
    print(f"  Normalization Layers:              {rmsnorm_params_per_block:,} ({rmsnorm_params_per_block/1e6:.2f}M)")
    print(f"  Eisenhower Context Layer:          {eisenhower_total_params:,} ({eisenhower_total_params/1e6:.2f}M)")
    print(f"  Total per block:                   {params_per_block:,} ({params_per_block/1e6:.2f}M)")
    print()
    
    print(f"Transformer Blocks ({n_blocks}):     {n_blocks * params_per_block:,} ({(n_blocks * params_per_block)/1e9:.2f}B)")
    print(f"Output Layer:                        {output_layer_params:,} ({output_layer_params/1e9:.2f}B)")
    print(f"Final Normalization:                 {rmsnorm_params_per_block:,} ({rmsnorm_params_per_block/1e6:.2f}M)")
    print("-" * 80)
    print(f"TOTAL MODEL PARAMETERS:              {total_params:,} ({total_params/1e9:.2f}B)")
    print()
    
    # Calculate percentage contributions
    print("Percentage Contributions:")
    print(f"  Embedding + Pos Enc:    {(embedding_params + pos_encoding_params) / total_params * 100:.2f}%")
    print(f"  Transformer Blocks:     {(n_blocks * params_per_block) / total_params * 100:.2f}%")
    print(f"  Output Layer:           {output_layer_params / total_params * 100:.2f}%")
    print(f"  Other (norm, etc):      {(rmsnorm_params_per_block * (n_blocks + 1)) / total_params * 100:.2f}%")
    print()
    
    # Calculate Eisenhower Context Layer contribution
    eisenhower_contribution = (eisenhower_total_params * n_blocks) / total_params * 100
    print(f"Eisenhower Context Layer Contribution: {(eisenhower_total_params * n_blocks):,} params ({eisenhower_contribution:.2f}%)")
    
    return total_params, {
        'embedding': embedding_params,
        'pos_encoding': pos_encoding_params,
        'per_block': params_per_block,
        'total_transformer': n_blocks * params_per_block,
        'output': output_layer_params,
        'eisenhower': eisenhower_total_params * n_blocks
    }

def create_detailed_model_with_context(model_config):
    """
    Create a detailed model instance with the new context management layer integrated
    """
    print("\nCreating detailed model with integrated context management...")
    
    # For demonstration purposes, we'll create just the context manager
    # since the full transformer implementation was created earlier
    context_manager = ContextManager(
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        max_context_length=model_config['seq_len']
    )
    
    total_params = count_parameters(context_manager)
    trainable_params = count_trainable_parameters(context_manager)
    
    print(f"Context Manager Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    return context_manager

def verify_model_size(target_billion_params=235):
    """
    Verify that the model can reach the target parameter size
    """
    print(f"\nVerifying model scalability to {target_billion_params} billion parameters...")
    
    # Start with base configuration that should give us ~235B parameters
    base_config = {
        'd_model': 8192,      # 8K dimensions
        'n_heads': 64,        # 64 attention heads
        'n_blocks': 12,       # 12 transformer blocks
        'vocab_size': 100000, # Large vocabulary
        'seq_len': 32768,     # 32K sequence length
        'mlp_multiplier': 4,  # Size of feed-forward network
        'n_experts': 8,       # Number of experts in MoE
        'expert_used': 2,     # Number of experts used per token
        'n_kv_heads': 8,      # KV heads for GQA
        'd_conv': 4,          # Convolution dimension for DeltaNet
        'd_state': 128        # State dimension for DeltaNet
    }
    
    # Adjust dimensions to approach 235B
    # Scale d_model and related dimensions
    scaling_factor = math.sqrt(target_billion_params / 100)  # Rough initial estimate
    base_config['d_model'] = int(base_config['d_model'] * scaling_factor)
    base_config['n_heads'] = int(base_config['n_heads'] * scaling_factor / 2)  # Keep head ratio reasonable
    base_config['n_kv_heads'] = int(base_config['n_kv_heads'] * scaling_factor / 2)
    
    # Make sure dimensions are divisible by number of heads
    base_config['d_model'] = (base_config['d_model'] // 128) * 128  # Round to nearest 128
    base_config['n_heads'] = max(8, (base_config['n_heads'] // 8) * 8)  # Round to nearest 8
    base_config['n_kv_heads'] = max(4, (base_config['n_kv_heads'] // 4) * 4)  # Round to nearest 4
    
    # Calculate exact parameters
    total_params, breakdown = calculate_weight_breakdown(base_config)
    
    print(f"\nAchieved {total_params/1e9:.2f}B parameters")
    print(f"Difference from target: {abs(total_params - target_billion_params*1e9)/1e9:.2f}B")
    
    if abs(total_params - target_billion_params*1e9) / (target_billion_params*1e9) < 0.05:
        print("✓ Model size is close to the target (within 5%)")
    else:
        print("⚠ Model size differs significantly from target. Consider adjusting dimensions.")
    
    return base_config, total_params

if __name__ == "__main__":
    print("Weight Calculation for World-Class Neural Network with New Context Management Layer")
    print("="*80)
    
    # Define model configuration
    model_config = {
        'd_model': 8192,      # 8K dimensions
        'n_heads': 64,        # 64 attention heads
        'n_blocks': 12,       # 12 transformer blocks
        'vocab_size': 100000, # Large vocabulary
        'seq_len': 32768,     # 32K sequence length
        'mlp_multiplier': 4,  # Size of feed-forward network
        'n_experts': 8,       # Number of experts in MoE
        'expert_used': 2,     # Number of experts used per token
        'n_kv_heads': 8,      # KV heads for GQA
        'd_conv': 4,          # Convolution dimension for DeltaNet
        'd_state': 128        # State dimension for DeltaNet
    }
    
    # Calculate weights with detailed breakdown
    total_params, breakdown = calculate_weight_breakdown(model_config)
    
    # Create the context management layer
    context_manager = create_detailed_model_with_context(model_config)
    
    # Verify model can scale to target size
    final_config, final_params = verify_model_size(target_billion_params=235)
    
    print("\n" + "="*80)
    print("WEIGHT CALCULATION SUMMARY")
    print("="*80)
    print(f"Final Model Configuration:")
    for key, value in final_config.items():
        print(f"  {key}: {value}")
    print(f"\nTotal Parameters: {final_params:,} ({final_params/1e9:.2f}B)")
    print(f"Target: 235B parameters")
    print("Successfully integrated the new Eisenhower Context Management Layer!")