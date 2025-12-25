import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import math

class EisenhowerContextLayer(nn.Module):
    """
    A specialized neural network layer that categorizes context based on the Eisenhower matrix:
    - Important + Urgent (Do First)
    - Important + Not Urgent (Schedule)
    - Unimportant + Urgent (Delegate)
    - Unimportant + Not Urgent (Eliminate)
    
    This layer processes input sequences and determines the importance and urgency of each token/segment.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super(EisenhowerContextLayer, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention for context analysis
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Separate networks for importance and urgency scoring
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.urgency_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Weight matrices for attention computation
        self.scaling_factor = math.sqrt(self.head_dim)
        
        # Final classification head for the four quadrants
        self.quadrant_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 4),  # Four quadrants
            nn.Softmax(dim=-1)
        )
        
        # Learnable parameters for context weighting
        self.context_weights = nn.Parameter(torch.ones(4))  # Weights for each quadrant
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through the Eisenhower Context Layer
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of:
                - Output tensor with context-aware representations
                - Dictionary with context analysis results
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V for attention-based context analysis
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention for context relevance
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling_factor
        
        if mask is not None:
            # Apply mask to attention scores
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to get context-relevant representations
        context_aware = torch.matmul(attention_probs, V)
        context_aware = context_aware.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        context_aware = self.out_proj(context_aware)
        
        # Calculate importance and urgency scores for each position
        importance_scores = self.importance_scorer(context_aware).squeeze(-1)  # (batch_size, seq_len)
        urgency_scores = self.urgency_scorer(context_aware).squeeze(-1)       # (batch_size, seq_len)
        
        # Classify each token into one of the four quadrants
        quadrant_inputs = torch.cat([
            context_aware,
            importance_scores.unsqueeze(-1).expand(-1, -1, 1),
            urgency_scores.unsqueeze(-1).expand(-1, -1, 1)
        ], dim=-1)
        
        quadrant_logits = self.quadrant_classifier(quadrant_inputs)  # (batch_size, seq_len, 4)
        
        # Calculate weighted outputs based on quadrant classification
        weighted_outputs = []
        for i in range(4):  # For each quadrant
            quadrant_mask = quadrant_logits[:, :, i].unsqueeze(-1)  # (batch_size, seq_len, 1)
            weighted_output = context_aware * quadrant_mask * self.context_weights[i]
            weighted_outputs.append(weighted_output)
        
        # Sum all quadrant-weighted outputs
        output = sum(weighted_outputs)
        
        # Prepare context analysis results
        context_analysis = {
            'importance_scores': importance_scores,
            'urgency_scores': urgency_scores,
            'quadrant_probabilities': quadrant_logits,
            'quadrant_weights': self.context_weights,
            'context_relevance_attention': attention_probs.mean(dim=1)  # Average over heads
        }
        
        return output, context_analysis


class ContextManager(nn.Module):
    """
    Complete context management system that integrates the Eisenhower layer
    with utilities for managing chat context and extracting important information.
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, max_context_length: int = 4096, dropout: float = 0.1):
        super(ContextManager, self).__init__()
        
        self.max_context_length = max_context_length
        self.eisenhower_layer = EisenhowerContextLayer(d_model, n_heads, dropout)
        
        # Additional processing for important information extraction
        self.important_token_extractor = nn.Linear(d_model, d_model)
        self.extraction_gate = nn.Linear(d_model, 1)
        
        # Buffer for maintaining context history
        self.register_buffer('context_memory', torch.zeros(1, max_context_length, d_model))
        self.register_buffer('context_positions', torch.arange(max_context_length).float())
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Process input with context management
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of processed tensor and context analysis
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply Eisenhower context layer
        processed_x, context_analysis = self.eisenhower_layer(x, mask)
        
        # Extract important tokens based on importance scores
        importance_scores = context_analysis['importance_scores']
        extraction_gates = torch.sigmoid(self.extraction_gate(processed_x)).squeeze(-1)
        
        # Combine importance and extraction gates
        combined_importance = importance_scores * extraction_gates
        
        # Apply selective processing based on importance
        important_mask = (combined_importance > 0.5).float().unsqueeze(-1)
        important_processed = self.important_token_extractor(processed_x * important_mask)
        
        # Combine important and regular processing
        output = processed_x + important_processed
        
        # Update context analysis with additional metrics
        context_analysis['combined_importance'] = combined_importance
        context_analysis['important_tokens_mask'] = important_mask.squeeze(-1)
        
        return output, context_analysis
    
    def extract_important_context(self, x: torch.Tensor, context_analysis: Dict, 
                                 top_k: int = 10) -> Tuple[List[int], torch.Tensor]:
        """
        Extract the most important tokens/segments from the context
        
        Args:
            x: Original input tensor
            context_analysis: Results from forward pass
            top_k: Number of most important items to extract
            
        Returns:
            Tuple of indices of important tokens and their embeddings
        """
        combined_importance = context_analysis['combined_importance']
        
        # Get top-k most important positions
        batch_size, seq_len = combined_importance.shape
        important_indices_batch = []
        important_embeddings_batch = []
        
        for i in range(batch_size):
            # Get top-k important positions for this sample
            _, top_indices = torch.topk(combined_importance[i], min(top_k, seq_len), largest=True, sorted=True)
            important_indices_batch.append(top_indices.tolist())
            important_embeddings_batch.append(x[i][top_indices])
        
        return important_indices_batch, torch.stack(important_embeddings_batch)


# Example usage and testing
if __name__ == "__main__":
    # Test the implementation
    batch_size = 2
    seq_len = 100
    d_model = 512
    
    # Create layer
    context_layer = ContextManager(d_model=d_model, n_heads=8)
    
    # Generate dummy input
    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.ones(batch_size, seq_len)  # All tokens are valid
    
    # Forward pass
    output, analysis = context_layer(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Importance scores shape: {analysis['importance_scores'].shape}")
    print(f"Urgency scores shape: {analysis['urgency_scores'].shape}")
    print(f"Quadrant probabilities shape: {analysis['quadrant_probabilities'].shape}")
    
    # Extract important context
    important_indices, important_embeddings = context_layer.extract_important_context(output, analysis, top_k=10)
    print(f"Number of important tokens extracted per sample: {[len(indices) for indices in important_indices]}")
    print(f"Important embeddings shape: {important_embeddings.shape}")
    
    print("Eisenhower Context Layer implementation completed successfully!")