"""
Layer-wise Inpainting Model
Channel concatenation approach with pretrained PixArt backbone

Uses UNet-style projections with skip connections:
- Input Projection: layers + mask → merged latent (4 channels)
- Output Projection: merged latent → layers (per-layer noise prediction)
"""

import torch
import torch.nn as nn
from diffusion.model.nets.PixArt import PixArt_XL_2
from diffusion.model.nets.projection_unet import ProjectionAutoencoder


class PixArtLayerInpainting(nn.Module):
    """
    Layer-wise inpainting using channel concatenation

    Input: (B, max_layers, 4, h, w) - VAE latents
           - Visible layers: clean latents
           - Masked layer: noisy latent
    Mask: (B, max_layers) - binary mask (1 for masked layer)

    Output: (B, max_layers, 4, h, w) - predicted noise for all layers
    """

    def __init__(
        self,
        pretrained_pixart=None,
        max_layers=6,
        input_size=32,
        pred_sigma=True,
        caption_channels=4096,
        model_max_length=120,
    ):
        super().__init__()
        self.max_layers = max_layers
        self.input_size = input_size

        # Input: max_layers * 4 (latents) + max_layers (mask)
        input_channels = max_layers * 4 + max_layers  # 6*4 + 6 = 30

        # UNet-style Input Projection: 30 → 4 channels
        # Deep architecture with skip connections and residual blocks
        self.input_proj = ProjectionAutoencoder(
            in_channels=input_channels,  # 30
            out_channels=4  # Merged latent space
        )

        # Pretrained PixArt backbone
        if pretrained_pixart is not None:
            print("[LayerInpainting] Using provided pretrained PixArt")
            self.pixart = pretrained_pixart
        else:
            print("[LayerInpainting] Creating PixArt from scratch")
            self.pixart = PixArt_XL_2(
                input_size=input_size,
                in_channels=4,
                caption_channels=caption_channels,
                model_max_length=model_max_length,
                pred_sigma=pred_sigma,
            )

        # UNet-style Output Projection: 4 → max_layers * 4
        # Deep architecture with skip connections and residual blocks
        output_channels = max_layers * 4  # 24
        self.output_proj = ProjectionAutoencoder(
            in_channels=4,  # Merged latent space
            out_channels=output_channels  # 24
        )

        # Store pred_sigma flag
        self.pred_sigma = pred_sigma

    def forward(self, layers, layer_mask, timestep, y, mask=None):
        """
        Args:
            layers: (B, max_layers, 4, h, w) - VAE latent space
                   - Visible layers: clean latents
                   - Masked layer: noisy latent (x_t at timestep t)
            layer_mask: (B, max_layers) - binary mask (1 for masked, 0 for visible)
            timestep: (B,) - diffusion timestep
            y: (B, 1, L, D) - text embeddings
            mask: (B, L) - text attention mask (optional)

        Returns:
            noise_pred: (B, max_layers, 4, h, w) - predicted noise
        """
        B, N, C, h, w = layers.shape
        assert N == self.max_layers, f"Expected {self.max_layers} layers, got {N}"

        # 1. Flatten layers: (B, N, 4, h, w) → (B, N*4, h, w)
        layers_flat = layers.reshape(B, N * C, h, w)

        # 2. Expand layer_mask to spatial dims: (B, N) → (B, N, h, w)
        mask_spatial = layer_mask.unsqueeze(-1).unsqueeze(-1).expand(B, N, h, w)

        # 3. Concatenate layers + mask: (B, N*4+N, h, w)
        input_with_mask = torch.cat([layers_flat, mask_spatial], dim=1)

        # 4. Project to 4 channels: (B, N*4+N, h, w) → (B, 4, h, w)
        x = self.input_proj(input_with_mask)

        # 5. Pretrained PixArt diffusion
        noise_pred_4ch = self.pixart(x, timestep, y, mask=mask)

        # Handle pred_sigma: output might be 8 channels (noise + variance)
        if self.pred_sigma and noise_pred_4ch.shape[1] == 8:
            noise_pred_4ch = noise_pred_4ch[:, :4]  # Take only noise prediction

        # 6. Project back to N*4 channels: (B, 4, h, w) → (B, N*4, h, w)
        noise_pred_flat = self.output_proj(noise_pred_4ch)

        # 7. Reshape to layer format: (B, N*4, h, w) → (B, N, 4, h, w)
        noise_pred = noise_pred_flat.reshape(B, N, C, h, w)

        return noise_pred

    @property
    def y_embedder(self):
        """Access y_embedder for null embedding"""
        return self.pixart.y_embedder

    @property
    def dtype(self):
        """Get model dtype"""
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for the underlying PixArt model"""
        if hasattr(self.pixart, 'enable_gradient_checkpointing'):
            self.pixart.enable_gradient_checkpointing()
        elif hasattr(self.pixart, 'gradient_checkpointing_enable'):
            self.pixart.gradient_checkpointing_enable()
        # If neither method exists, gradient checkpointing is not supported
        # Don't manually implement it as it can cause recursion issues


def load_pretrained_pixart(checkpoint_path, input_size=32):
    """Load pretrained PixArt model"""
    from diffusion.model.nets.PixArt import PixArt_XL_2

    model = PixArt_XL_2(
        input_size=input_size,
        in_channels=4,
        caption_channels=4096,
        model_max_length=120,
        pred_sigma=True,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

    # Remove 'module.' prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Pretrained PixArt] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    return model
