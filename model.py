import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from tqdm import tqdm


# --------------- Diffusion Model Base Utilities ---------------

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


# --------------- Beta Schedule Utilities ---------------

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# --------------- U-Net Components ---------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2's spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(skip_channels * 2, out_channels)
        self.attn = AttentionGate(F_g=skip_channels, F_l=skip_channels, F_int=out_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.attn(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# --------------- UNet Model for CPM and IRM ---------------

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, attention=False, pretrain=False, light=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = attention
        self.pretrain = pretrain
        self.light = light
        factor = 2 if bilinear else 1

        if pretrain:
            # Use pretrained ResNet34 encoder
            import torchvision.models as models
            from torchvision.models import ResNet34_Weights
            
            resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            
            if n_channels != 3:
                self.input_layer = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.input_layer.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True).repeat(1, n_channels, 1, 1)
            else:
                self.input_layer = resnet.conv1
            
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  # 64 channels
            self.layer2 = resnet.layer2  # 128 channels
            self.layer3 = resnet.layer3  # 256 channels
            self.layer4 = resnet.layer4  # 512 channels

            # Decoder
            if self.attention:
                self.up1 = AttentionUp(512, 256, 256, bilinear)
                self.up2 = AttentionUp(256, 128, 128, bilinear)
                self.up3 = AttentionUp(128, 64, 64, bilinear)
                self.up4 = AttentionUp(64, 64, 64, bilinear)
            else:
                self.up1 = Up(512, 256, 256, bilinear)
                self.up2 = Up(256, 128, 128, bilinear)
                self.up3 = Up(128, 64, 64, bilinear)
                self.up4 = Up(64, 64, 64, bilinear)
        else:
            # Define channels based on light factor
            if self.light == 1:
                # Lighter version
                self.inc = DoubleConv(n_channels, 16)
                self.down1 = Down(16, 32)
                self.down2 = Down(32, 64)
                self.down3 = Down(64, 128)
                self.down4 = Down(128, 256 // factor)
                
                if self.attention:
                    self.up1 = AttentionUp(256, 128, 128 // factor, bilinear)
                    self.up2 = AttentionUp(128, 64, 64 // factor, bilinear)
                    self.up3 = AttentionUp(64, 32, 32 // factor, bilinear)
                    self.up4 = AttentionUp(32, 16, 16, bilinear)
                else:
                    self.up1 = Up(256, 128, 128 // factor, bilinear)
                    self.up2 = Up(128, 64, 64 // factor, bilinear)
                    self.up3 = Up(64, 32, 32 // factor, bilinear)
                    self.up4 = Up(32, 16, 16, bilinear)
            elif self.light == 2:
                # Even lighter version
                self.inc = DoubleConv(n_channels, 8)
                self.down1 = Down(8, 16)
                self.down2 = Down(16, 32)
                self.down3 = Down(32, 64)
                self.down4 = Down(64, 128 // factor)
                
                if self.attention:
                    self.up1 = AttentionUp(128, 64, 64 // factor, bilinear)
                    self.up2 = AttentionUp(64, 32, 32 // factor, bilinear)
                    self.up3 = AttentionUp(32, 16, 16 // factor, bilinear)
                    self.up4 = AttentionUp(16, 8, 8, bilinear)
                else:
                    self.up1 = Up(128, 64, 64 // factor, bilinear)
                    self.up2 = Up(64, 32, 32 // factor, bilinear)
                    self.up3 = Up(32, 16, 16 // factor, bilinear)
                    self.up4 = Up(16, 8, 8, bilinear)
            else:
                # Standard UNet
                self.inc = DoubleConv(n_channels, 64)
                self.down1 = Down(64, 128)
                self.down2 = Down(128, 256)
                self.down3 = Down(256, 512)
                self.down4 = Down(512, 1024 // factor)
                
                if self.attention:
                    self.up1 = AttentionUp(1024, 512, 512 // factor, bilinear)
                    self.up2 = AttentionUp(512, 256, 256 // factor, bilinear)
                    self.up3 = AttentionUp(256, 128, 128 // factor, bilinear)
                    self.up4 = AttentionUp(128, 64, 64, bilinear)
                else:
                    self.up1 = Up(1024, 512, 512 // factor, bilinear)
                    self.up2 = Up(512, 256, 256 // factor, bilinear)
                    self.up3 = Up(256, 128, 128 // factor, bilinear)
                    self.up4 = Up(128, 64, 64, bilinear)
        
        # Output convolution
        if light == 1:
            self.outc = OutConv(16, n_classes)
        elif light == 2:
            self.outc = OutConv(8, n_classes)
        else:
            self.outc = OutConv(64, n_classes)
            
        self.use_residual = True

    def forward(self, x, time_emb=None):
        x_in = x  # Store input for residual connection
        
        if self.pretrain:
            x = self.input_layer(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x1 = x  # 64 channels
            x2 = self.layer1(x1)  # 64 channels
            x3 = self.layer2(x2)  # 128 channels
            x4 = self.layer3(x3)  # 256 channels
            x5 = self.layer4(x4)  # 512 channels
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

        # Skip connections in the decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        # Residual connection
        if self.use_residual:
            if x.shape != x_in.shape:
                x_in = F.interpolate(x_in, size=x.shape[2:], mode='bilinear', align_corners=True)
            return x + x_in
        else:
            return x


# --------------- Diffusion Model ---------------

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        timesteps=1000,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02,
        device='cuda'
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        self.device = device
        
        # Set up beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Move betas to the correct device
        betas = betas.to(device)
        
        self.register_buffer('betas', betas)
        
        # Pre-compute diffusion parameters
        alphas = 1. - betas
        self.register_buffer('alphas', alphas)
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Log calculation clipped for numerical stability
        # Create minimum value tensor on the same device
        min_val = torch.tensor(1e-20, device=device)
        posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, min_val))
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)
        
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        
        # Loss function
        self.loss_func = nn.L1Loss(reduction='mean')

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        """
        noise = default(noise, lambda: torch.randn_like(x_start, device=self.device))
        
        # Extract parameters for the given timestep
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        # Forward process
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, condition, t, noise=None):
        """
        Training loss calculation
        """
        noise = default(noise, lambda: torch.randn_like(x_start, device=self.device))
        
        # Add noise to get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        
        # Predict the noise
        predicted_noise = self.denoise_fn(torch.cat([condition, x_noisy], dim=1), t)
        
        # Calculate loss
        loss = self.loss_func(noise, predicted_noise)
        
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and the predicted noise
        """
        sqrt_recip_alphas_cumprod_t = extract(1. / torch.sqrt(self.alphas_cumprod), t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract(torch.sqrt(1. / self.alphas_cumprod - 1), t, x_t.shape)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x_t, condition, t, t_index):
        """
        Single reverse step: p(x_{t-1} | x_t)
        """
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = extract(torch.rsqrt(self.alphas), t, x_t.shape)
        
        # Predict noise
        predicted_noise = self.denoise_fn(torch.cat([condition, x_t], dim=1), t)
        
        # Predict x_0
        pred_x0 = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        # Clip x_0 prediction for stability
        pred_x0 = torch.clamp(pred_x0, -1., 1.)
        
        # Compute mean for p(x_{t-1} | x_t, x_0)
        posterior_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return pred_x0
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t, device=self.device)
            return posterior_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, condition, num_steps=100):
        """
        Sample from the model using num_steps
        """
        batch_size = condition.shape[0]
        shape = condition.shape
        
        # Start from pure noise
        img = torch.randn(shape, device=self.device)
        
        # Timesteps to sample at
        if num_steps < self.timesteps:
            step_indices = torch.linspace(0, self.timesteps-1, num_steps, dtype=torch.long, device=self.device)
        else:
            step_indices = torch.arange(self.timesteps, device=self.device)
        
        step_indices = step_indices.flip(0)  # Reverse for sampling from noise to clean
        
        for i, step in enumerate(step_indices):
            # Convert to batch of timesteps
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            img = self.p_sample(img, condition, t, i)
            
        return img
    
    def forward(self, x_start, condition, t=None):
        """
        Forward pass used during training
        """
        b, c, h, w = x_start.shape
        
        # Sample random timesteps
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        
        # Calculate loss
        return self.p_losses(x_start, condition, t)


# --------------- Coarse-to-Fine Diffusion Model ---------------

class CoarseToFineDiffusion(nn.Module):
    def __init__(
        self,
        cpm_type='unet',
        attention=False,
        pretrain=False,
        light=0,
        diffusion_steps=1000,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02,
        device='cuda'
    ):
        super().__init__()
        
        # Coarse Prediction Module (CPM)
        if cpm_type == 'unet':
            self.cpm = UNet(
                n_channels=1,
                n_classes=1,
                bilinear=False,
                attention=attention,
                pretrain=pretrain,
                light=light
            )
        elif cpm_type == 'resnet':
            # Simple ResNet implementation for CPM
            self.cpm = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                self._make_residual_block(64, 64, 2),
                self._make_residual_block(64, 128, 2, stride=2),
                self._make_residual_block(128, 256, 2, stride=2),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(64, 1, kernel_size=3, padding=1)
            )
        elif cpm_type == 'lighter':
            # Lighter UNet for CPM
            self.cpm = UNet(
                n_channels=1,
                n_classes=1,
                bilinear=False,
                attention=False,
                pretrain=False,
                light=max(1, light)  # Always use at least light=1
            )
        
        # Iterative Refinement Module (IRM) - Diffusion Model
        # Input to the diffusion model is the concatenation of incomplete sinogram and CPM output
        irm_unet = UNet(
            n_channels=2,  # Condition (1) + Noise (1)
            n_classes=1,   # Predicted noise
            bilinear=False,
            attention=attention,
            pretrain=False,
            light=light
        )
        
        self.irm = GaussianDiffusion(
            denoise_fn=irm_unet,
            timesteps=diffusion_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        
        self.device = device
        
    def _make_residual_block(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._residual_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._residual_block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _residual_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            downsample if downsample else nn.Identity()
        )
    
    def forward(self, incomplete_sinogram, complete_sinogram=None, t=None):
        """
        Forward pass during training
        """
        # Get coarse prediction
        coarse_pred = self.cpm(incomplete_sinogram)
        
        if complete_sinogram is not None:
            # Training mode
            # Calculate residual between ground truth and coarse prediction
            residual = complete_sinogram - coarse_pred
            
            # Train the diffusion model to predict this residual
            loss = self.irm(residual, incomplete_sinogram, t)
            
            return coarse_pred, loss
        else:
            # Inference mode (no ground truth)
            return coarse_pred
    
    @torch.no_grad()
    def sample(self, incomplete_sinogram, num_steps=100):
        """
        Sample from the model during inference
        """
        # Get coarse prediction
        coarse_pred = self.cpm(incomplete_sinogram)
        
        # Use diffusion model to predict the residual
        predicted_residual = self.irm.sample(incomplete_sinogram, num_steps)
        
        # Add the predicted residual to the coarse prediction
        refined_prediction = coarse_pred + predicted_residual
        
        return coarse_pred, predicted_residual, refined_prediction