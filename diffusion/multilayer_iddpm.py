import torch
import torch.nn.functional as F


class MultiLayerIDDPM:
    """
    Wrapper around IDDPM for multi-layer diffusion training.
    Handles (B, N, C, H, W) tensors and applies layer masking.
    """
    
    def __init__(self, iddpm, scale_factor=0.18215):
        self.iddpm = iddpm
        self.scale_factor = scale_factor
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, layer_mask=None):
        """
        Compute training loss for multi-layer generation.
        
        Args:
            model: MultiLayerPixArt model
            x_start: (B, N, C, H, W) - clean multi-layer latents
            t: (B,) - timesteps
            model_kwargs: dict with 'y', 'mask' for text conditioning
            noise: optional noise tensor
            layer_mask: (B, N) - mask for valid layers (1=valid, 0=padding)
        
        Returns:
            dict with 'loss' tensor of shape (B,)
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        B, N, C, H, W = x_start.shape
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise using IDDPM's schedule
        # q_sample expects: sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * noise
        x_t = self._q_sample(x_start, t, noise)
        
        # Forward through model
        # model expects: (B, N, C, H, W), timestep, y, mask
        model_output = model(x_t, t, **model_kwargs)
        
        # model_output: (B, N, C_out, H, W) where C_out = C or 2*C (if pred_sigma)
        C_out = model_output.shape[2]
        
        if C_out == 2 * C:
            # Split prediction and variance
            model_pred, model_var = model_output.split(C, dim=2)
        else:
            model_pred = model_output
            model_var = None
        
        # Target is noise (epsilon prediction)
        target = noise
        
        # MSE loss: (B, N, C, H, W)
        loss = F.mse_loss(model_pred, target, reduction='none')
        
        # Reduce over C, H, W: (B, N)
        loss = loss.mean(dim=[2, 3, 4])
        
        # Apply layer mask if provided
        if layer_mask is not None:
            # Mask out padded layers
            loss = loss * layer_mask
            # Average over valid layers only
            num_valid = layer_mask.sum(dim=1).clamp(min=1)
            loss = loss.sum(dim=1) / num_valid  # (B,)
        else:
            loss = loss.mean(dim=1)  # (B,)
        
        return {"loss": loss}
    
    def _q_sample(self, x_start, t, noise):
        """
        Add noise to x_start at timestep t.
        Uses IDDPM's alpha schedule.
        """
        # Get alpha_bar values from IDDPM
        # IDDPM stores: sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
        sqrt_alpha_bar = self._extract(self.iddpm.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.iddpm.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
    
    def _extract(self, arr, timesteps, broadcast_shape):
        """
        Extract values from array at timesteps and broadcast to shape.
        """
        res = arr.to(timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)
