import torch
import torch.nn.functional as F
import numpy as np


class ReferenceIDDPM:
    def __init__(self, iddpm):
        self.iddpm = iddpm
    
    def training_losses(self, model, x, t, model_kwargs=None):
        """x: (B, C, H, W) 타겟 레이어"""
        if model_kwargs is None:
            model_kwargs = {}
        
        noise = torch.randn_like(x)
        x_t = self._q_sample(x, t, noise)
        
        # ReferencePixArt forward 호출
        model_output = model(
            x_target=x_t,
            timestep=t,
            y=model_kwargs['y'],
            x_ref=model_kwargs['x_ref'],
            mask=model_kwargs.get('mask'),
        )
        
        # Epsilon prediction loss
        if model_output.shape[1] == 2 * x.shape[1]:
            model_pred, _ = model_output.chunk(2, dim=1)
        else:
            model_pred = model_output
        
        loss = F.mse_loss(model_pred, noise, reduction='none')
        loss = loss.mean(dim=[1, 2, 3])  # (B,)
        
        return {"loss": loss}
    
    def _q_sample(self, x_start, t, noise):
        sqrt_alpha = self._extract(self.iddpm.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._extract(self.iddpm.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise
    
    def _extract(self, arr, timesteps, shape):
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        res = arr.to(timesteps.device)[timesteps].float()
        while len(res.shape) < len(shape):
            res = res.unsqueeze(-1)
        return res.expand(shape)