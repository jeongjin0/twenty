import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class ReferenceIDDPM:
    def __init__(self, iddpm):
        self.iddpm = iddpm
        self.num_timesteps = 100
    
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
        """DDIM Sampling"""
        # Build alpha schedule
        alphas_cumprod = self.iddpm.alphas_cumprod
        if isinstance(alphas_cumprod, np.ndarray):
            alphas_cumprod = torch.from_numpy(alphas_cumprod)
        alphas_cumprod = alphas_cumprod.to(device)
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Timestep sequence
        timesteps = torch.linspace(self.num_timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)
        
        # Get null_y for CFG
        null_y = model.y_embedder.y_embedding.unsqueeze(0).unsqueeze(0).to(device).to(y.dtype)
        
        for i in tqdm(range(steps), desc="DDIM Sampling", leave=False):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t.expand(shape[0])

            # CFG: conditional (text + ref) vs unconditional (no text, no ref)
            # Conditional: text + reference
            out_cond = model(
                x_target=x,
                timestep=t_batch,
                y=y,
                x_ref=x_ref,
                mask=None,
            )

            # Unconditional: no text, no reference
            out_uncond = model.forward_without_ref(
                x_target=x,
                timestep=t_batch,
                y=null_y.expand(y.shape[0], -1, -1, -1),
                mask=None,
            )
            
            in_channels = shape[1]
            if out_cond.shape[1] == 2 * in_channels:
                eps_cond = out_cond[:, :in_channels]
                eps_uncond = out_uncond[:, :in_channels]
            else:
                eps_cond = out_cond
                eps_uncond = out_uncond
            
            noise_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            
            # DDIM update
            alpha_t = alphas_cumprod[t]
            alpha_t_next = alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred
        
        return x
    
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
