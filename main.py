# main.py
import os
import argparse
import torch
import logging
import random
import numpy as np
import datetime
from pathlib import Path

import train
import evaluate
from model import CoarseToFineDiffusion

def setup_logging(log_dir):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('main')
    return logger

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Coarse-to-Fine Diffusion Model for Sinogram Restoration')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode: train or test')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='2e9div', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='c2f_diffusion', help='Model name')
    parser.add_argument('--cpm_type', type=str, default='unet', choices=['unet', 'resnet', 'lighter'], help='Type of CPM')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in UNet')
    parser.add_argument('--use_pretrained', action='store_true', help='Use pretrained encoder for UNet')
    parser.add_argument('--light_factor', type=int, default=0, choices=[0, 1, 2], help='Light factor for UNet (0=standard, 1=lighter, 2=lightest)')
    
    # Diffusion model settings
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--sampling_steps', type=int, default=100, help='Number of sampling steps')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='Beta schedule type')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='Beta start value')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end value')
    
    # Output settings
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Base directory for logs')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (epochs)')
    parser.add_argument('--save_freq', type=int, default=5, help='Checkpoint saving frequency (epochs)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    models_dir = os.path.join(args.models_dir, timestamp)
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting run with timestamp: {timestamp}")
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Set up device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = CoarseToFineDiffusion(
        cpm_type=args.cpm_type,
        attention=args.use_attention,
        pretrain=args.use_pretrained,
        light=args.light_factor,
        diffusion_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device
    )
    model = model.to(device)
    
    # Save model architecture summary
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Write parameters to log directory
    with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    # Run training or evaluation
    if args.mode == 'train':
        train.train(
            model=model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            models_dir=models_dir,
            log_dir=log_dir,
            resume=args.resume,
            sampling_steps=args.sampling_steps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq
        )
    else:
        evaluate.evaluate(
            model=model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            log_dir=log_dir,
            checkpoint_path=args.resume,
            sampling_steps=args.sampling_steps
        )

if __name__ == '__main__':
    main()