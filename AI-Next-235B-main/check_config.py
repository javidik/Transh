#!/usr/bin/env python3
"""
Script to verify the training configuration before starting the training process
"""

import os
import sys
import torch
import subprocess

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("Warning: Python 3.8 or higher is recommended")
    else:
        print(f"Python version: {sys.version.split()[0]} - OK")

def check_cuda_availability():
    """Check CUDA availability"""
    print("\nChecking CUDA availability...")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA available: No")
        print("Warning: Training will run on CPU which will be very slow")

def check_disk_space():
    """Check available disk space"""
    print("\nChecking disk space...")
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        print(f"Free disk space: {free_gb} GB")
        if free_gb < 500:  # Less than 500GB
            print("Warning: Less than 500GB free space. Consider getting more space for checkpoints and data.")
        else:
            print("Disk space - OK")
    except Exception as e:
        print(f"Could not check disk space: {e}")

def check_memory():
    """Check system memory"""
    print("\nChecking system memory...")
    try:
        mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_gib = mem_bytes / (1024.**3)
        print(f"Total system memory: {mem_gib:.2f} GB")
        if mem_gib < 128:  # Less than 128GB
            print("Warning: Less than 128GB RAM. For 235B parameter model, more memory is recommended.")
        else:
            print("System memory - OK")
    except Exception as e:
        print(f"Could not check system memory: {e}")

def check_packages():
    """Check required packages"""
    print("\nChecking required packages...")
    required_packages = [
        'torch', 'transformers', 'datasets', 'tqdm', 'numpy', 'deepspeed'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {package}: OK")
        except ImportError:
            missing_packages.append(package)
            print(f"  {package}: MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: bash setup_train.sh to install them")
        return False
    
    return True

def check_data_directory():
    """Check if data directory exists and has files"""
    print("\nChecking data directory...")
    data_dir = "/workspace/pile_data"
    
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        if files:
            print(f"Data directory exists with {len(files)} files - OK")
            return True
        else:
            print("Data directory exists but is empty")
            print("Note: Sample data was created during setup, but for real training, download The Pile dataset")
            return True
    else:
        print(f"Data directory does not exist: {data_dir}")
        print("Run: mkdir -p /workspace/pile_data")
        return False

def check_model_parameters():
    """Check if model can be initialized with current parameters"""
    print("\nChecking model parameters...")
    try:
        from nn_architecture import TransformerArchitecture
        
        # Create a smaller model for testing initialization
        model = TransformerArchitecture(
            vocab_size=50257,
            d_model=1024,  # Smaller for test
            nhead=8,
            num_layers=4,  # Smaller for test
            dim_feedforward=4096,
            max_seq_len=1024,
            dropout=0.1,
            num_experts=16,  # Smaller for test
            active_experts=4,
            context_dim=256
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Test model parameter count: {param_count:,}")
        print("Model initialization - OK")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def main():
    print("Verifying training configuration...\n")
    
    all_checks_passed = True
    
    # Run all checks
    check_python_version()
    check_cuda_availability()
    check_disk_space()
    check_memory()
    all_checks_passed &= check_packages()
    all_checks_passed &= check_data_directory()
    all_checks_passed &= check_model_parameters()
    
    print("\n" + "="*50)
    if all_checks_passed:
        print("All checks passed! Ready to start training.")
        print("\nTo begin training, run:")
        print("  python train_pile.py")
        print("\nFor distributed training with DeepSpeed:")
        print("  deepspeed --num_gpus=8 train_pile.py --deepspeed ds_config.json")
    else:
        print("Some checks failed. Please address the issues above before starting training.")
    print("="*50)

if __name__ == "__main__":
    import shutil  # Import here to avoid error if not available
    main()