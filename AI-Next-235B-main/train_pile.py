"""
Training script for the world-class neural network on The Pile dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
import math
from tqdm import tqdm
import json
import logging
from typing import Optional, Tuple, List
import glob
import mmap

# Import our neural network architecture
from nn_architecture import TransformerArchitecture

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PileDataset(Dataset):
    """
    Dataset class for loading and processing The Pile dataset
    """
    def __init__(self, data_dir: str, tokenizer_name: str = "gpt2", max_length: int = 2048):
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get list of .jsonl files in the data directory
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
        logger.info(f"Found {len(self.files)} files in {data_dir}")
        
        # Calculate total number of samples
        self.total_samples = 0
        for file_path in self.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    self.total_samples += 1
        
        logger.info(f"Total samples in dataset: {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which file contains the sample
        current_idx = 0
        for file_path in self.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if current_idx == idx:
                        sample = json.loads(line)
                        
                        # Extract text (could be in different keys depending on pile category)
                        text = ""
                        if isinstance(sample, dict):
                            if 'text' in sample:
                                text = sample['text']
                            elif 'meta' in sample:
                                text = str(sample.get('meta', ''))
                            else:
                                # Try to find any text-like field
                                for key, value in sample.items():
                                    if isinstance(value, str):
                                        text = value
                                        break
                        
                        # Tokenize the text
                        tokens = self.tokenizer.encode(
                            text,
                            max_length=self.max_length,
                            truncation=True,
                            padding='max_length',
                            return_tensors=None
                        )
                        
                        # Create input and target sequences
                        input_ids = torch.tensor(tokens, dtype=torch.long)
                        
                        # Shift target sequence (for next-token prediction)
                        target_ids = torch.cat([input_ids[1:], torch.tensor([self.tokenizer.pad_token_id])])
                        
                        return {
                            'input_ids': input_ids,
                            'target_ids': target_ids,
                            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
                        }
                    
                    current_idx += 1
        
        raise IndexError(f"Index {idx} out of range")

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    """Create dataloader with appropriate settings for large models"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

class Trainer:
    """
    Main training class for the neural network
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        accumulation_steps: int = 4,
        save_dir: str = "./checkpoints",
        log_interval: int = 10
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training metrics
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(enumerate(self.train_dataloader), total=num_batches)
        
        for batch_idx, batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Normalize loss for gradient accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            total_loss += loss.item() * self.accumulation_steps
            
            # Log progress
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': avg_loss,
                    'lr': lr,
                    'step': self.global_step
                })
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss}")
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("best_model")
        
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, f"{name}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, f"{name}.pth")
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

def get_learning_rate_schedule(optimizer, num_training_steps, warmup_steps):
    """Create learning rate schedule"""
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

def main():
    # Configuration
    pile_data_dir = "/workspace/pile_data"  # Path to The Pile dataset
    batch_size = 4  # Reduced for memory efficiency, increase based on GPU memory
    accumulation_steps = 8  # Effective batch size = batch_size * accumulation_steps
    effective_batch_size = batch_size * accumulation_steps  # 32
    learning_rate = 1e-4
    weight_decay = 0.1
    epochs = 3
    max_length = 2048  # Sequence length
    warmup_ratio = 0.01  # 1% of training steps for warmup
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransformerArchitecture(
        vocab_size=50257,  # GPT-2 vocab size
        d_model=8192,      # Hidden dimension (adjusted for 235B params)
        nhead=64,          # Number of attention heads
        num_layers=101,    # Total number of layers
        dim_feedforward=32768,  # FFN dimension
        max_seq_len=max_length,
        dropout=0.1,
        num_experts=256,   # Number of experts per MoE layer (as per requirements)
        active_experts=42, # Active experts per token (except special ones)
        context_dim=512    # Dimension for context management
    ).to(device)
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = PileDataset(pile_data_dir, max_length=max_length)
    
    # Split dataset into train and validation
    train_size = int(0.95 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataloader = get_dataloader(train_subset, batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_subset, batch_size, shuffle=False)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_dataloader) * epochs // accumulation_steps
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    scheduler = get_learning_rate_schedule(optimizer, num_training_steps, num_warmup_steps)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        accumulation_steps=accumulation_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        # Save checkpoint
        trainer.save_checkpoint(f"epoch_{epoch+1}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()