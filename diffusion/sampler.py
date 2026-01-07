"""
DDIM Sampler for inference
"""
import torch
from tqdm import tqdm


class SimpleDDIMSampler:
    """간단한 DDIM Sampler"""
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    @torch.no_grad()
    def sample(self, model, shape, y, y_mask, x_ref, null_y, null_mask, cfg_scale=4.5, steps=20, device='cuda'):
        """DDIM Sampling with reference

        Args:
            null_y: Unconditional embedding from T5("") - must match training!
            null_mask: Attention mask for null_y
        """
        self.alphas_cumprod = self.alphas_cumprod.to(device)

        x = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, steps + 1, dtype=torch.long, device=device)

        for i in tqdm(range(steps), desc="Sampling"):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t.expand(shape[0])

            # CFG
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t_batch, t_batch], dim=0)

            # Use T5("") embedding instead of model.y_embedder.y_embedding
            # This matches training where empty string "" was encoded by T5
            y_in = torch.cat([y, null_y], dim=0)

            x_ref_in = torch.cat([x_ref, x_ref], dim=0)

            if y_mask is not None:
                mask_in = torch.cat([y_mask, null_mask], dim=0)
            else:
                mask_in = None

            # Call model with references
            noise_pred = model(x_in, t_in, y_in, x_ref_in, mask=mask_in)


            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            # pred_sigma=True면 출력이 2*in_channels (noise + sigma)
            in_channels = shape[1]
            if noise_pred_cond.shape[1] == in_channels * 2:
                noise_pred_cond = noise_pred_cond[:, :in_channels]
                noise_pred_uncond = noise_pred_uncond[:, :in_channels]

            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

            alpha_t = self.alphas_cumprod[t]
            alpha_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_t_next) * x0_pred + torch.sqrt(1 - alpha_t_next) * noise_pred

        return x
