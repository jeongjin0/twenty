import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusion.sampler import SimpleDDIMSampler


class ReferenceIDDPM:
    def __init__(self, iddpm):
        self.iddpm = iddpm
        self.num_timesteps = getattr(iddpm, 'original_num_steps', 1000)
    
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


    @torch.no_grad()
    def ddim_sample(self, model, shape, y, x_ref, mask=None, steps=20, cfg_scale=4.5, device='cuda'):
        """DDIM Sampling using SimpleDDIMSampler"""
        sampler = SimpleDDIMSampler(num_timesteps=self.num_timesteps)
        return sampler.sample(
            model=model,
            shape=shape,
            y=y,
            y_mask=mask,
            x_ref=x_ref,
            cfg_scale=cfg_scale,
            steps=steps,
            device=device
        )
    
    @torch.no_grad()
    def evaluate(self, model, vae, z_target, z_ref, y, mask=None, steps=20, cfg_scale=4.5, scale_factor=0.18215):
        """
        Training 중 evaluate: 실제 z_ref로 z_target을 생성하고 비교
        
        Args:
            z_target: (B, 5, h, w) - ground truth target latent
            z_ref: (B, N_ref, 5, h, w) - reference latents
            y: (B, 1, L, D) - text embedding
        
        Returns:
            dict with generated/gt images and metrics
        """
        device = z_target.device
        B, C, h, w = z_target.shape
        
        # Generate
        z_gen = self.ddim_sample(
            model=model,
            shape=(B, C, h, w),
            y=y,
            x_ref=z_ref,
            mask=mask,
            steps=steps,
            cfg_scale=cfg_scale,
            device=device,
        )

        # Decode both (RGB only, channels 0-3)
        z_gen_rgb = z_gen[:, :4] / scale_factor
        z_target_rgb = z_target[:, :4] / scale_factor
        z_ref_rgb = z_ref[:, :, :4] / scale_factor  # (B, N_ref, 4, h, w)

        img_gen = vae.decode(z_gen_rgb).sample  # (B, 3, H, W)
        img_target = vae.decode(z_target_rgb).sample

        # Decode refs
        B, N_ref, _, h, w = z_ref_rgb.shape
        z_ref_flat = z_ref_rgb.reshape(B * N_ref, 4, h, w)
        img_ref_flat = vae.decode(z_ref_flat).sample
        img_ref = img_ref_flat.reshape(B, N_ref, 3, img_ref_flat.shape[-2], img_ref_flat.shape[-1])

        # Compute metrics
        mse = F.mse_loss(img_gen, img_target).item()

        return {
            'img_gen': img_gen,      # (B, 3, H, W)
            'img_target': img_target,  # (B, 3, H, W)
            'img_ref': img_ref,      # (B, N_ref, 3, H, W)
            'z_gen': z_gen,
            'z_target': z_target,
            'mse': mse,
        }
