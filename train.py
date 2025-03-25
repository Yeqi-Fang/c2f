import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from dataset import SinogramDataset, create_dataloaders

logger = logging.getLogger('train')

def save_checkpoint(model, optimizer, scaler, epoch, step, models_dir, name):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(models_dir, f"{name}_epoch{epoch}_step{step}.pth")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path

def save_visualization(incomplete, coarse_pred, refined, complete, step, log_dir, epoch):
    """Save visualization of model predictions"""
    os.makedirs(os.path.join(log_dir, 'visualizations'), exist_ok=True)
    vis_path = os.path.join(log_dir, 'visualizations', f'epoch_{epoch}_step_{step}.png')
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    # Process up to 4 samples from the batch
    for i in range(min(4, incomplete.shape[0])):
        # Get images
        inc = incomplete[i, 0].cpu().numpy()
        coarse = coarse_pred[i, 0].cpu().numpy()
        ref = refined[i, 0].cpu().numpy()
        comp = complete[i, 0].cpu().numpy()
        
        # Plot
        axes[i, 0].imshow(inc, cmap='magma')
        axes[i, 0].set_title('Incomplete')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(coarse, cmap='magma')
        axes[i, 1].set_title('Coarse Prediction')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(ref, cmap='magma')
        axes[i, 2].set_title('Refined Prediction')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(comp, cmap='magma')
        axes[i, 3].set_title('Ground Truth')
        axes[i, 3].axis('off')
    
    # Add title and save
    plt.suptitle(f'Epoch {epoch}, Step {step}')
    plt.tight_layout()
    plt.savefig(vis_path, dpi=200)
    plt.close(fig)
    
    logger.info(f"Visualization saved to {vis_path}")
    return vis_path

def compute_metrics(pred, target, win_size=7):
    """Compute evaluation metrics using scikit-image for each sample and average them.
    
    Args:
        pred (torch.Tensor): Predicted images of shape (B, 1, H, W).
        target (torch.Tensor): Ground truth images of shape (B, 1, H, W).
        win_size (int): Window size for SSIM. Must be an odd integer less than or equal to the image dimensions.
    
    Returns:
        dict: Dictionary with averaged 'mse', 'psnr', and 'ssim' values.
    """
    # Move to CPU and convert to numpy arrays
    pred_np = pred.detach().cpu().numpy()   # shape: (B, 1, H, W)
    target_np = target.detach().cpu().numpy()  # shape: (B, 1, H, W)
    
    batch_size = pred_np.shape[0]
    mse_total, psnr_total, ssim_total = 0.0, 0.0, 0.0

    for i in range(batch_size):
        # Remove the channel dimension: now each image is (H, W)
        pred_img = pred_np[i, 0]
        target_img = target_np[i, 0]
        
        # Mean Squared Error for the sample
        mse_sample = np.mean((pred_img - target_img) ** 2)
        mse_total += mse_sample
        
        # Determine data_range based on ground truth
        data_range = target_img.max() - target_img.min()
        
        # Compute PSNR and SSIM for the sample
        psnr_sample = compare_psnr(target_img, pred_img, data_range=data_range)
        ssim_sample = compare_ssim(target_img, pred_img, data_range=data_range, win_size=win_size)
        psnr_total += psnr_sample
        ssim_total += ssim_sample

    # Average over the batch
    mse_avg = mse_total / batch_size
    psnr_avg = psnr_total / batch_size
    ssim_avg = ssim_total / batch_size
    
    return {'mse': mse_avg, 'psnr': psnr_avg, 'ssim': ssim_avg}


def train(
    model,
    data_dir,
    batch_size,
    num_workers,
    num_epochs,
    lr,
    weight_decay,
    device,
    models_dir,
    log_dir,
    resume=None,
    sampling_steps=100,
    eval_freq=1,
    save_freq=5
):
    """Main training function"""
    logger.info(f"Starting training with {num_epochs} epochs")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Set up gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Initialize training variables
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if provided
    if resume is not None and os.path.isfile(resume):
        logger.info(f"Loading checkpoint from {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('step', 0)
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        epoch_coarse_loss = 0.0
        epoch_refine_loss = 0.0
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (incomplete, complete) in enumerate(pbar):
            # Move data to device
            incomplete = incomplete.to(device)
            complete = complete.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                # Get coarse prediction and diffusion loss
                coarse_pred, refine_loss = model(incomplete, complete)
                
                # Coarse prediction loss
                coarse_loss = nn.MSELoss()(coarse_pred, complete)
                
                # Total loss
                total_loss = coarse_loss + refine_loss
            
            # Backward pass with mixed precision
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress bar
            global_step += 1
            epoch_loss += total_loss.item()
            epoch_coarse_loss += coarse_loss.item()
            epoch_refine_loss += refine_loss.item()
            
            pbar.set_postfix({
                'loss': total_loss.item(),
                'coarse_loss': coarse_loss.item(),
                'refine_loss': refine_loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Save visualizations periodically
            if global_step % 500 == 0 or batch_idx == len(train_loader) - 1:
                model.eval()
                with torch.no_grad():
                    # Sample from the model
                    coarse_pred, _, refined_pred = model.sample(incomplete, sampling_steps)
                
                # Save visualization
                save_visualization(
                    incomplete=incomplete,
                    coarse_pred=coarse_pred,
                    refined=refined_pred,
                    complete=complete,
                    step=global_step,
                    log_dir=log_dir,
                    epoch=epoch+1
                )
                model.train()
        
        # Log epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        avg_coarse_loss = epoch_coarse_loss / len(train_loader)
        avg_refine_loss = epoch_refine_loss / len(train_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Loss: {avg_loss:.6f}, "
                   f"Coarse Loss: {avg_coarse_loss:.6f}, "
                   f"Refine Loss: {avg_refine_loss:.6f}")
        
        # Validation
        if (epoch + 1) % eval_freq == 0:
            val_loss, val_metrics = validate(model, val_loader, device, sampling_steps, log_dir, epoch+1)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, epoch+1, global_step, models_dir, "best")
                logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % save_freq == 0:
            save_checkpoint(model, optimizer, scaler, epoch+1, global_step, models_dir, f"epoch{epoch+1}")
    
    # Save final model
    save_checkpoint(model, optimizer, scaler, num_epochs, global_step, models_dir, "final")
    logger.info("Training completed!")
    
    return model

def validate(model, val_loader, device, sampling_steps, log_dir, epoch):
    """Validate the model"""
    logger.info("Starting validation...")
    model.eval()
    
    val_loss = 0.0
    metrics_coarse = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}
    metrics_refined = {'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0}
    
    # Sample a few batches for visualization
    vis_samples = []
    
    with torch.no_grad():
        for batch_idx, (incomplete, complete) in enumerate(tqdm(val_loader, desc="Validation")):
            # Move data to device
            incomplete = incomplete.to(device)
            complete = complete.to(device)
            
            # Get coarse prediction and diffusion loss
            coarse_pred, refine_loss = model(incomplete, complete)
            
            # Sample from the model
            coarse_sample, residual, refined_pred = model.sample(incomplete, sampling_steps)
            
            # Compute metrics
            batch_metrics_coarse = compute_metrics(coarse_sample, complete)
            batch_metrics_refined = compute_metrics(refined_pred, complete)
            
            # Update metrics
            for k in metrics_coarse:
                metrics_coarse[k] += batch_metrics_coarse[k]
                metrics_refined[k] += batch_metrics_refined[k]
            
            # Calculate loss
            coarse_loss = nn.MSELoss()(coarse_pred, complete)
            total_loss = coarse_loss + refine_loss
            val_loss += total_loss.item()
            
            # Store a few samples for visualization
            if len(vis_samples) < 2 and batch_idx % 5 == 0:
                vis_samples.append((incomplete, coarse_sample, residual, refined_pred, complete))
    
    # Average metrics
    val_loss /= len(val_loader)
    for k in metrics_coarse:
        metrics_coarse[k] /= len(val_loader)
        metrics_refined[k] /= len(val_loader)
    
    # Log metrics
    logger.info(f"Validation - Loss: {val_loss:.6f}")
    logger.info(f"Coarse Metrics - MSE: {metrics_coarse['mse']:.6f}, PSNR: {metrics_coarse['psnr']:.4f} dB, SSIM: {metrics_coarse['ssim']:.4f}")
    logger.info(f"Refined Metrics - MSE: {metrics_refined['mse']:.6f}, PSNR: {metrics_refined['psnr']:.4f} dB, SSIM: {metrics_refined['ssim']:.4f}")
    
    # Save validation visualizations
    if vis_samples:
        save_validation_visualizations(vis_samples, log_dir, epoch)
    
    model.train()
    return val_loss, {'coarse': metrics_coarse, 'refined': metrics_refined}

def save_validation_visualizations(vis_samples, log_dir, epoch):
    """Save visualizations from validation"""
    os.makedirs(os.path.join(log_dir, 'validation'), exist_ok=True)
    vis_path = os.path.join(log_dir, 'validation', f'epoch_{epoch}_validation.png')
    
    num_samples = len(vis_samples)
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i, (incomplete, coarse, residual, refined, complete) in enumerate(vis_samples):
        # Get the first image in batch
        inc = incomplete[0, 0].cpu().numpy()
        crs = coarse[0, 0].cpu().numpy()
        res = residual[0, 0].cpu().numpy()
        ref = refined[0, 0].cpu().numpy()
        comp = complete[0, 0].cpu().numpy()
        
        # Plot
        axes[i, 0].imshow(inc, cmap='magma')
        axes[i, 0].set_title('Incomplete')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(crs, cmap='magma')
        axes[i, 1].set_title('Coarse Prediction')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(res, cmap='magma')
        axes[i, 2].set_title('Predicted Residual')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(ref, cmap='magma')
        axes[i, 3].set_title('Refined Prediction')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(comp, cmap='magma')
        axes[i, 4].set_title('Ground Truth')
        axes[i, 4].axis('off')
    
    plt.suptitle(f'Validation Samples - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(vis_path, dpi=200)
    plt.close(fig)
    
    logger.info(f"Validation visualization saved to {vis_path}")
