"""
UNet-style Projection Autoencoder with Skip Connections

Used for both:
1. Projection pretraining (merge/decompose strategy)
2. Main layer inpainting model (layer projection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation

    Architecture:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> (+residual) -> SiLU
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class ProjectionAutoencoder(nn.Module):
    """
    UNet-style Projection Autoencoder with Skip Connections

    Architecture:
    - Encoder: in_channels → 64 → 128 → 256 → 4
    - Decoder: 4 → 256 → 128 → 64 → out_channels (with 3 skip connections)

    Key features:
    - 4-channel bottleneck (compatible with PixArt latent space)
    - Skip connections at 64, 128, 256 channels
    - Residual blocks for stable deep training
    - Much deeper than simple conv layers (~9+ effective layers)

    Args:
        in_channels: Input channels (e.g., 24 for 6 layers × 4 channels)
        out_channels: Output channels (e.g., 24 for 6 layers × 4 channels)
    """

    def __init__(self, in_channels=24, out_channels=24):
        super().__init__()

        # ========== Encoder ==========
        # Stage 1: in_channels → 64
        self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc_norm1 = nn.GroupNorm(8, 64)
        self.enc_res1 = ResidualBlock(64)

        # Stage 2: 64 → 128
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.enc_norm2 = nn.GroupNorm(8, 128)
        self.enc_res2 = ResidualBlock(128)

        # Stage 3: 128 → 256
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.enc_norm3 = nn.GroupNorm(8, 256)
        self.enc_res3 = ResidualBlock(256)

        # Bottleneck: 256 → 4
        self.enc_final = nn.Conv2d(256, 4, kernel_size=1)

        # ========== Decoder ==========
        # Bottleneck: 4 → 256
        self.dec_conv1 = nn.Conv2d(4, 256, kernel_size=1)
        self.dec_norm1 = nn.GroupNorm(8, 256)
        self.dec_res1 = ResidualBlock(256)

        # Stage 3: 256+256 (skip) → 128
        self.dec_conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.dec_norm2 = nn.GroupNorm(8, 128)
        self.dec_res2 = ResidualBlock(128)

        # Stage 2: 128+128 (skip) → 64
        self.dec_conv3 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.dec_norm3 = nn.GroupNorm(8, 64)
        self.dec_res3 = ResidualBlock(64)

        # Stage 1: 64+64 (skip) → out_channels
        self.dec_final = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def encode(self, x):
        """
        Encode to 4-channel bottleneck with skip connections

        Returns:
            z: (B, 4, h, w) - bottleneck
            skips: List of skip connection features [skip1_64, skip2_128, skip3_256]
        """
        # Stage 1: → 64
        x1 = F.silu(self.enc_norm1(self.enc_conv1(x)))
        x1 = self.enc_res1(x1)
        skip1 = x1  # (B, 64, h, w)

        # Stage 2: → 128
        x2 = F.silu(self.enc_norm2(self.enc_conv2(x1)))
        x2 = self.enc_res2(x2)
        skip2 = x2  # (B, 128, h, w)

        # Stage 3: → 256
        x3 = F.silu(self.enc_norm3(self.enc_conv3(x2)))
        x3 = self.enc_res3(x3)
        skip3 = x3  # (B, 256, h, w)

        # Bottleneck: → 4
        z = self.enc_final(x3)  # (B, 4, h, w)

        return z, [skip1, skip2, skip3]

    def decode(self, z, skips):
        """
        Decode from 4-channel bottleneck with skip connections

        Args:
            z: (B, 4, h, w) - bottleneck
            skips: List of skip connection features [skip1_64, skip2_128, skip3_256]

        Returns:
            x: (B, out_channels, h, w) - reconstruction
        """
        skip1, skip2, skip3 = skips

        # Bottleneck: 4 → 256
        x = F.silu(self.dec_norm1(self.dec_conv1(z)))
        x = self.dec_res1(x)  # (B, 256, h, w)

        # Stage 3: 256+256 (skip) → 128
        x = torch.cat([x, skip3], dim=1)  # (B, 512, h, w)
        x = F.silu(self.dec_norm2(self.dec_conv2(x)))
        x = self.dec_res2(x)  # (B, 128, h, w)

        # Stage 2: 128+128 (skip) → 64
        x = torch.cat([x, skip2], dim=1)  # (B, 256, h, w)
        x = F.silu(self.dec_norm3(self.dec_conv3(x)))
        x = self.dec_res3(x)  # (B, 64, h, w)

        # Stage 1: 64+64 (skip) → out_channels
        x = torch.cat([x, skip1], dim=1)  # (B, 128, h, w)
        x = self.dec_final(x)  # (B, out_channels, h, w)

        return x

    def forward(self, x):
        """
        Full forward pass: encode → decode

        Args:
            x: (B, in_channels, h, w)

        Returns:
            recon: (B, out_channels, h, w)
        """
        z, skips = self.encode(x)
        recon = self.decode(z, skips)
        return recon

    def get_bottleneck(self, x):
        """Get 4-channel bottleneck representation"""
        z, _ = self.encode(x)
        return z


class InputProjection(nn.Module):
    """
    Input Projection: Multiple layers → Single merged latent

    Architecture: UNet-style encoder that produces 4-channel bottleneck
    """
    def __init__(self, max_layers=6):
        super().__init__()
        in_channels = max_layers * 4  # e.g., 6 layers × 4 channels = 24
        self.autoencoder = ProjectionAutoencoder(
            in_channels=in_channels,
            out_channels=4  # Output 4 channels (merged latent)
        )

    def forward(self, layers):
        """
        Args:
            layers: (B, N, 4, h, w) where N = max_layers

        Returns:
            merged: (B, 4, h, w)
        """
        B, N, C, h, w = layers.shape
        layers_flat = layers.reshape(B, N * C, h, w)  # (B, N*4, h, w)
        merged = self.autoencoder(layers_flat)  # (B, 4, h, w)
        return merged


class OutputProjection(nn.Module):
    """
    Output Projection: Single merged latent → Multiple layers

    Architecture: UNet-style decoder that produces N×4 channels
    """
    def __init__(self, max_layers=6):
        super().__init__()
        out_channels = max_layers * 4  # e.g., 6 layers × 4 channels = 24
        self.autoencoder = ProjectionAutoencoder(
            in_channels=4,  # Input 4 channels (merged latent)
            out_channels=out_channels
        )
        self.max_layers = max_layers

    def forward(self, merged):
        """
        Args:
            merged: (B, 4, h, w)

        Returns:
            layers: (B, N, 4, h, w) where N = max_layers
        """
        B, C, h, w = merged.shape
        layers_flat = self.autoencoder(merged)  # (B, N*4, h, w)
        layers = layers_flat.reshape(B, self.max_layers, 4, h, w)
        return layers
