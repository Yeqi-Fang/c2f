import os
import logging
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from dataset import create_dataloaders

logger = logging.getLogger('evaluate')

def evaluate(
    model,
    data_dir,
    batch_size,
    num_workers,
    device,
    log_dir,
    checkpoint_path=None,
    sampling_steps=100
):
    """Comprehensive evaluation of the model"""
    logger.info("Starting evaluation...")
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model.eval()
    model = model.to(device)
    
    # Create dataloader for test set
    _, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    logger.info(f"Created test dataloader with {len(test_loader)} batches")
    
    # Create evaluation directory
    os.makedirs(os.path.join(log_dir, 'evaluation'), exist_ok=True)
    
    # Metrics storage
    metrics = {
        'coarse': {'mse': [], 'psnr': [], 'ssim': []},
        'refined': {'mse': [], 'psnr': [], 'ssim': []}
    }
    
    # Sample storage for detailed examination
    samples = []
    
    with torch.no_grad():
        for batch_idx, (incomplete, complete) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Move data to device
            incomplete = incomplete.to(device)
            complete = complete.to(device)
            
            # Sample from the model
            coarse_pred, residual, refined_pred = model.sample(incomplete, sampling_steps)
            
            # Convert to CPU and numpy for metrics calculation
            incomplete_np = incomplete.cpu().numpy()
            coarse_pred_np = coarse_pred.cpu().numpy()
            residual_np = residual.cpu().numpy()
            refined_pred_np = refined_pred.cpu().numpy()
            complete_np = complete.cpu().numpy()
            
            # Calculate metrics for each sample in batch
            for i in range(incomplete.shape[0]):
                # Get single images
                inc_img = incomplete_np[i, 0]
                coarse_img = coarse_pred_np[i, 0]
                residual_img = residual_np[i, 0]
                refined_img = refined_pred_np[i, 0]
                complete_img = complete_np[i, 0]
                
                # Calculate MSE
                coarse_mse = np.mean((coarse_img - complete_img) ** 2)
                refined_mse = np.mean((refined_img - complete_img) ** 2)
                
                # Calculate PSNR
                coarse_psnr = peak_signal_noise_ratio(complete_img, coarse_img, data_range=complete_img.max() - complete_img.min())
                refined_psnr = peak_signal_noise_ratio(complete_img, refined_img, data_range=complete_img.max() - complete_img.min())
                
                # Calculate SSIM
                coarse_ssim = structural_similarity(complete_img, coarse_img, data_range=complete_img.max() - complete_img.min())
                refined_ssim = structural_similarity(complete_img, refined_img, data_range=complete_img.max() - complete_img.min())
                
                # Store metrics
                metrics['coarse']['mse'].append(coarse_mse)
                metrics['coarse']['psnr'].append(coarse_psnr)
                metrics['coarse']['ssim'].append(coarse_ssim)
                
                metrics['refined']['mse'].append(refined_mse)
                metrics['refined']['psnr'].append(refined_psnr)
                metrics['refined']['ssim'].append(refined_ssim)
                
                # Store samples for visualization (first 50 only)
                if len(samples) < 50:
                    samples.append({
                        'incomplete': inc_img,
                        'coarse': coarse_img,
                        'residual': residual_img,
                        'refined': refined_img,
                        'complete': complete_img,
                        'metrics': {
                            'coarse': {'mse': coarse_mse, 'psnr': coarse_psnr, 'ssim': coarse_ssim},
                            'refined': {'mse': refined_mse, 'psnr': refined_psnr, 'ssim': refined_ssim}
                        }
                    })
    
    # Calculate average metrics
    avg_metrics = {
        'coarse': {k: np.mean(v) for k, v in metrics['coarse'].items()},
        'refined': {k: np.mean(v) for k, v in metrics['refined'].items()}
    }
    
    # Calculate standard deviation
    std_metrics = {
        'coarse': {k: np.std(v) for k, v in metrics['coarse'].items()},
        'refined': {k: np.std(v) for k, v in metrics['refined'].items()}
    }
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(f"Coarse Prediction - MSE: {avg_metrics['coarse']['mse']:.6f} (±{std_metrics['coarse']['mse']:.6f})")
    logger.info(f"Coarse Prediction - PSNR: {avg_metrics['coarse']['psnr']:.4f} dB (±{std_metrics['coarse']['psnr']:.4f})")
    logger.info(f"Coarse Prediction - SSIM: {avg_metrics['coarse']['ssim']:.4f} (±{std_metrics['coarse']['ssim']:.4f})")
    
    logger.info(f"Refined Prediction - MSE: {avg_metrics['refined']['mse']:.6f} (±{std_metrics['refined']['mse']:.6f})")
    logger.info(f"Refined Prediction - PSNR: {avg_metrics['refined']['psnr']:.4f} dB (±{std_metrics['refined']['psnr']:.4f})")
    logger.info(f"Refined Prediction - SSIM: {avg_metrics['refined']['ssim']:.4f} (±{std_metrics['refined']['ssim']:.4f})")
    
    # Improvement calculation
    psnr_improvement = avg_metrics['refined']['psnr'] - avg_metrics['coarse']['psnr']
    ssim_improvement = avg_metrics['refined']['ssim'] - avg_metrics['coarse']['ssim']
    mse_reduction = avg_metrics['coarse']['mse'] - avg_metrics['refined']['mse']
    
    logger.info(f"Improvements - PSNR: +{psnr_improvement:.4f} dB, SSIM: +{ssim_improvement:.4f}, MSE: -{mse_reduction:.6f}")
    
    # Save metrics to file
    metrics_file = os.path.join(log_dir, 'evaluation', 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Evaluation Results:\n")
        f.write(f"Coarse Prediction - MSE: {avg_metrics['coarse']['mse']:.6f} (±{std_metrics['coarse']['mse']:.6f})\n")
        f.write(f"Coarse Prediction - PSNR: {avg_metrics['coarse']['psnr']:.4f} dB (±{std_metrics['coarse']['psnr']:.4f})\n")
        f.write(f"Coarse Prediction - SSIM: {avg_metrics['coarse']['ssim']:.4f} (±{std_metrics['coarse']['ssim']:.4f})\n\n")
        
        f.write(f"Refined Prediction - MSE: {avg_metrics['refined']['mse']:.6f} (±{std_metrics['refined']['mse']:.6f})\n")
        f.write(f"Refined Prediction - PSNR: {avg_metrics['refined']['psnr']:.4f} dB (±{std_metrics['refined']['psnr']:.4f})\n")
        f.write(f"Refined Prediction - SSIM: {avg_metrics['refined']['ssim']:.4f} (±{std_metrics['refined']['ssim']:.4f})\n\n")
        
        f.write(f"Improvements - PSNR: +{psnr_improvement:.4f} dB, SSIM: +{ssim_improvement:.4f}, MSE: -{mse_reduction:.6f}\n")
    
    # Save visualizations
    save_sample_visualizations(samples, log_dir)
    
    # Create histograms of metrics
    create_metric_histograms(metrics, log_dir)
    
    logger.info(f"Evaluation completed. Results saved to {metrics_file}")
    
    return avg_metrics, std_metrics, samples

def save_sample_visualizations(samples, log_dir):
    """Save visualizations of sample predictions"""
    logger.info("Saving sample visualizations...")
    
    # Create directory
    vis_dir = os.path.join(log_dir, 'evaluation', 'samples')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Sort samples by refined PSNR for better visualization
    sorted_samples = sorted(samples, key=lambda x: x['metrics']['refined']['psnr'], reverse=True)
    
    # Save individual samples
    for i, sample in enumerate(sorted_samples):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # Get min/max for consistent colormap
        vmin = min(sample['incomplete'].min(), sample['coarse'].min(), 
                  sample['residual'].min(), sample['refined'].min(), sample['complete'].min())
        vmax = max(sample['incomplete'].max(), sample['coarse'].max(), 
                  sample['residual'].max(), sample['refined'].max(), sample['complete'].max())
        
        # Plot images
        axes[0].imshow(sample['incomplete'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[0].set_title('Incomplete')
        axes[0].axis('off')
        
        axes[1].imshow(sample['coarse'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Coarse\nPSNR: {sample['metrics']['coarse']['psnr']:.2f} dB\nSSIM: {sample['metrics']['coarse']['ssim']:.4f}")
        axes[1].axis('off')
        
        axes[2].imshow(sample['residual'], cmap='magma')
        axes[2].set_title('Residual')
        axes[2].axis('off')
        
        axes[3].imshow(sample['refined'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[3].set_title(f"Refined\nPSNR: {sample['metrics']['refined']['psnr']:.2f} dB\nSSIM: {sample['metrics']['refined']['ssim']:.4f}")
        axes[3].axis('off')
        
        axes[4].imshow(sample['complete'], cmap='magma', vmin=vmin, vmax=vmax)
        axes[4].set_title('Ground Truth')
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'sample_{i+1:03d}.png'), dpi=150)
        plt.close(fig)
    
    # Create grid of best and worst samples
    create_sample_grid(sorted_samples[:9], os.path.join(vis_dir, 'best_samples_grid.png'), "Best Samples by PSNR")
    create_sample_grid(sorted_samples[-9:], os.path.join(vis_dir, 'worst_samples_grid.png'), "Worst Samples by PSNR")
    
    logger.info(f"Saved {len(samples)} sample visualizations to {vis_dir}")

def create_sample_grid(samples, save_path, title):
    """Create a grid visualization of samples"""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    for i, sample in enumerate(samples):
        row, col = i // 3, i % 3
        
        # Show refined prediction with PSNR
        im = axes[row, col].imshow(sample['refined'], cmap='magma')
        axes[row, col].set_title(f"PSNR: {sample['metrics']['refined']['psnr']:.2f} dB")
        axes[row, col].axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

def create_metric_histograms(metrics, log_dir):
    """Create histograms of metrics"""
    logger.info("Creating metric histograms...")
    
    # Create directory
    hist_dir = os.path.join(log_dir, 'evaluation', 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    
    # Create histograms
    for metric_name in ['mse', 'psnr', 'ssim']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        ax.hist(metrics['coarse'][metric_name], bins=30, alpha=0.5, label='Coarse')
        ax.hist(metrics['refined'][metric_name], bins=30, alpha=0.5, label='Refined')
        
        # Add vertical lines for means
        ax.axvline(np.mean(metrics['coarse'][metric_name]), color='blue', linestyle='dashed', linewidth=2, label=f'Coarse Mean: {np.mean(metrics["coarse"][metric_name]):.4f}')
        ax.axvline(np.mean(metrics['refined'][metric_name]), color='orange', linestyle='dashed', linewidth=2, label=f'Refined Mean: {np.mean(metrics["refined"][metric_name]):.4f}')
        
        # Set title and labels
        metric_title = metric_name.upper() if metric_name == 'mse' else metric_name.upper()
        ax.set_title(f'Distribution of {metric_title} Values')
        ax.set_xlabel(metric_name.upper())
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, f'{metric_name}_histogram.png'), dpi=150)
        plt.close(fig)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # PSNR scatter plot
    axes[0].scatter(metrics['coarse']['psnr'], metrics['refined']['psnr'], alpha=0.5)
    axes[0].plot([0, 50], [0, 50], 'r--')  # Diagonal line
    axes[0].set_xlabel('Coarse PSNR (dB)')
    axes[0].set_ylabel('Refined PSNR (dB)')
    axes[0].set_title('PSNR Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # SSIM scatter plot
    axes[1].scatter(metrics['coarse']['ssim'], metrics['refined']['ssim'], alpha=0.5)
    axes[1].plot([0, 1], [0, 1], 'r--')  # Diagonal line
    axes[1].set_xlabel('Coarse SSIM')
    axes[1].set_ylabel('Refined SSIM')
    axes[1].set_title('SSIM Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # Improvement histogram
    psnr_improvements = np.array(metrics['refined']['psnr']) - np.array(metrics['coarse']['psnr'])
    axes[2].hist(psnr_improvements, bins=30)
    axes[2].axvline(np.mean(psnr_improvements), color='red', linestyle='dashed', 
                   label=f'Mean: {np.mean(psnr_improvements):.4f} dB')
    axes[2].set_xlabel('PSNR Improvement (dB)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('PSNR Improvement Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, 'metrics_comparison.png'), dpi=150)
    plt.close(fig)
    
    logger.info(f"Saved metric histograms to {hist_dir}")