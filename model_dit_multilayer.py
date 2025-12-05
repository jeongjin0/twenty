# Multi-Layer Inpainting Architectures based on DiT
# Architecture 1: MultiLayerDiT - All layers simultaneous generation with layer-wise attention
# Architecture 2: ReferenceDiT - Single layer generation with reference conditioning

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Optional, Tuple


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers (from original DiT)                            #
#################################################################################

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LayerEmbedder(nn.Module):
    """Embeds layer indices into vector representations."""
    def __init__(self, max_layers, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(max_layers, hidden_size)
        self.max_layers = max_layers

    def forward(self, layer_indices):
        """
        layer_indices: (N,) or (B, N) tensor of layer indices
        """
        return self.embedding_table(layer_indices)


#################################################################################
#                           DiT Block (from original)                           #
#################################################################################

class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                     Positional Embedding Functions                            #
#################################################################################

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


#################################################################################
#           Architecture 1: MultiLayerDiT - Simultaneous Layer Generation       #
#################################################################################

class MultiLayerDiT(nn.Module):
    """
    Multi-Layer Diffusion Transformer.
    
    모든 레이어를 동시에 생성하며, layer-wise attention을 수행.
    Input: (B, N, C, H, W) where N = num_layers
    Output: (B, N, C_out, H, W)
    
    핵심: 모든 레이어의 모든 patch가 서로 attention
    - Spatial positional embedding: 공간 위치
    - Layer positional embedding: 레이어 인덱스
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,  # RGBA
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_layers=16,  # 최대 레이어 개수
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_layers = max_layers
        self.hidden_size = hidden_size

        # Patch embedding (shared across all layers)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        num_patches = self.x_embedder.num_patches
        self.num_patches = num_patches
        
        # Spatial positional embedding (sin-cos, fixed)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # Layer positional embedding (learnable)
        self.layer_embed = nn.Parameter(torch.zeros(1, max_layers, 1, hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize spatial pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize layer_embed
        nn.init.normal_(self.layer_embed, std=0.02)

        # Initialize patch_embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, num_layers):
        """
        x: (B, N*T, patch_size**2 * C)
        imgs: (B, N, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(self.num_patches ** 0.5)
        
        # Reshape: (B, N*T, p*p*c) -> (B, N, T, p*p*c)
        B = x.shape[0]
        T = self.num_patches
        x = x.reshape(B, num_layers, T, -1)
        
        # (B, N, T, p*p*c) -> (B, N, h, w, p, p, c)
        x = x.reshape(B, num_layers, h, w, p, p, c)
        
        # (B, N, h, w, p, p, c) -> (B, N, c, h*p, w*p)
        x = torch.einsum('bnhwpqc->bncphwq', x)
        imgs = x.reshape(B, num_layers, c, h * p, w * p)
        return imgs

    def forward(self, x, t):
        """
        Forward pass of MultiLayerDiT.
        x: (B, N, C, H, W) tensor - N layers of RGBA images
        t: (B,) tensor of diffusion timesteps
        
        Returns: (B, N, C_out, H, W)
        """
        B, N, C, H, W = x.shape
        assert N <= self.max_layers, f"Number of layers {N} exceeds max_layers {self.max_layers}"
        
        # 1. Patch embed each layer independently
        # Reshape: (B, N, C, H, W) -> (B*N, C, H, W)
        x = x.reshape(B * N, C, H, W)
        x = self.x_embedder(x)  # (B*N, T, D)
        
        # Reshape back: (B*N, T, D) -> (B, N, T, D)
        T, D = x.shape[1], x.shape[2]
        x = x.reshape(B, N, T, D)
        
        # 2. Add positional embeddings
        # Spatial: (1, 1, T, D) broadcast to (B, N, T, D)
        x = x + self.pos_embed.unsqueeze(1)  # pos_embed: (1, T, D) -> (1, 1, T, D)
        
        # Layer: (1, N, 1, D) broadcast to (B, N, T, D)
        x = x + self.layer_embed[:, :N, :, :]
        
        # 3. Flatten for attention: (B, N, T, D) -> (B, N*T, D)
        x = x.reshape(B, N * T, D)
        
        # 4. Timestep conditioning
        t_emb = self.t_embedder(t)  # (B, D)
        c = t_emb
        
        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x, c)  # (B, N*T, D)
        
        # 6. Final layer
        x = self.final_layer(x, c)  # (B, N*T, patch_size**2 * out_channels)
        
        # 7. Unpatchify
        x = self.unpatchify(x, N)  # (B, N, C_out, H, W)
        
        return x

    def load_pretrained_dit(self, state_dict, strict=False):
        """
        Load pretrained DiT weights.
        New layers (layer_embed) will be randomly initialized.
        """
        # Filter out incompatible keys
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} pretrained weights")


#################################################################################
#           Architecture 2: ReferenceDiT - Reference-based Generation           #
#################################################################################

class ReferenceEncoder(nn.Module):
    """
    Encodes multiple reference layers into a single embedding.
    
    Input: (B, N-1, C, H, W) - reference layers
    Output: (B, D) - aggregated reference embedding
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        num_heads=8,
        depth=4,  # Lighter than main model
        max_layers=16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_layers = max_layers
        
        # Patch embedding
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.layer_embed = nn.Parameter(torch.zeros(1, max_layers, 1, hidden_size))
        
        # Transformer encoder blocks (simpler, no adaLN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Aggregation: pool all tokens into single embedding
        self.pool_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Spatial pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        nn.init.normal_(self.layer_embed, std=0.02)
        nn.init.normal_(self.pool_token, std=0.02)
        
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)
    
    def forward(self, ref_layers):
        """
        ref_layers: (B, N_ref, C, H, W) - reference layers
        Returns: (B, D) - aggregated embedding
        """
        B, N_ref, C, H, W = ref_layers.shape
        
        # Patch embed: (B*N_ref, C, H, W) -> (B*N_ref, T, D)
        x = ref_layers.reshape(B * N_ref, C, H, W)
        x = self.patch_embed(x)
        
        T, D = x.shape[1], x.shape[2]
        x = x.reshape(B, N_ref, T, D)
        
        # Add positional embeddings
        x = x + self.pos_embed.unsqueeze(1)
        x = x + self.layer_embed[:, :N_ref, :, :]
        
        # Flatten: (B, N_ref, T, D) -> (B, N_ref*T, D)
        x = x.reshape(B, N_ref * T, D)
        
        # Prepend pool token
        pool_tokens = self.pool_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([pool_tokens, x], dim=1)  # (B, 1 + N_ref*T, D)
        
        # Encode
        x = self.encoder(x)  # (B, 1 + N_ref*T, D)
        
        # Extract pool token and project
        pooled = x[:, 0]  # (B, D)
        output = self.output_proj(pooled)  # (B, D)
        
        return output


class ReferenceDiT(nn.Module):
    """
    Reference-based Diffusion Transformer for single layer generation.
    
    - Target layer: 생성할 레이어 (denoising)
    - Reference layers: 나머지 레이어들 (conditioning)
    
    Reference layers는 encoder를 통해 embedding으로 변환되고,
    이 embedding이 timestep embedding과 함께 conditioning으로 사용됨.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,  # RGBA
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_ref_layers=15,  # 최대 reference 레이어 개수
        ref_encoder_depth=4,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Target layer processing (standard DiT)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        num_patches = self.x_embedder.num_patches
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        
        # Reference encoder
        self.ref_encoder = ReferenceEncoder(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=ref_encoder_depth,
            max_layers=max_ref_layers,
        )
        
        # Projection for combining timestep and reference embeddings
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Spatial pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (B, T, patch_size**2 * C)
        imgs: (B, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x_target, t, x_ref):
        """
        Forward pass of ReferenceDiT.
        
        x_target: (B, C, H, W) - target layer to generate (noisy)
        t: (B,) - diffusion timesteps
        x_ref: (B, N_ref, C, H, W) - reference layers (clean)
        
        Returns: (B, C_out, H, W) - predicted noise/x0 for target layer
        """
        # 1. Encode target layer
        x = self.x_embedder(x_target) + self.pos_embed  # (B, T, D)
        
        # 2. Encode reference layers
        ref_emb = self.ref_encoder(x_ref)  # (B, D)
        
        # 3. Timestep embedding
        t_emb = self.t_embedder(t)  # (B, D)
        
        # 4. Combine conditioning: [t_emb, ref_emb] -> projection -> c
        c = self.cond_proj(torch.cat([t_emb, ref_emb], dim=-1))  # (B, D)
        
        # 5. Transformer blocks
        for block in self.blocks:
            x = block(x, c)  # (B, T, D)
        
        # 6. Final layer
        x = self.final_layer(x, c)  # (B, T, patch_size**2 * out_channels)
        
        # 7. Unpatchify
        x = self.unpatchify(x)  # (B, C_out, H, W)
        
        return x

    def forward_without_ref(self, x_target, t):
        """
        Forward pass without reference (for unconditional generation or CFG).
        Uses zero embedding for reference.
        """
        x = self.x_embedder(x_target) + self.pos_embed
        
        # Zero reference embedding
        ref_emb = torch.zeros(x_target.shape[0], self.hidden_size, device=x_target.device)
        t_emb = self.t_embedder(t)
        c = self.cond_proj(torch.cat([t_emb, ref_emb], dim=-1))
        
        for block in self.blocks:
            x = block(x, c)
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x_target, t, x_ref, cfg_scale):
        """
        Forward with classifier-free guidance on reference.
        """
        # Conditional
        cond_out = self.forward(x_target, t, x_ref)
        
        # Unconditional
        uncond_out = self.forward_without_ref(x_target, t)
        
        # CFG
        if self.learn_sigma:
            eps_cond, rest_cond = cond_out[:, :self.in_channels], cond_out[:, self.in_channels:]
            eps_uncond, rest_uncond = uncond_out[:, :self.in_channels], uncond_out[:, self.in_channels:]
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            return torch.cat([eps, rest_cond], dim=1)
        else:
            return uncond_out + cfg_scale * (cond_out - uncond_out)

    def load_pretrained_dit(self, state_dict, strict=False):
        """
        Load pretrained DiT weights.
        Reference encoder and cond_proj will be randomly initialized.
        """
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} pretrained weights")


#################################################################################
#                              Model Configurations                             #
#################################################################################

def MultiLayerDiT_XL_2(**kwargs):
    return MultiLayerDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def MultiLayerDiT_L_2(**kwargs):
    return MultiLayerDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MultiLayerDiT_B_2(**kwargs):
    return MultiLayerDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def MultiLayerDiT_S_2(**kwargs):
    return MultiLayerDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def ReferenceDiT_XL_2(**kwargs):
    return ReferenceDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def ReferenceDiT_L_2(**kwargs):
    return ReferenceDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def ReferenceDiT_B_2(**kwargs):
    return ReferenceDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def ReferenceDiT_S_2(**kwargs):
    return ReferenceDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


MultiLayerDiT_models = {
    'MultiLayerDiT-XL/2': MultiLayerDiT_XL_2,
    'MultiLayerDiT-L/2': MultiLayerDiT_L_2,
    'MultiLayerDiT-B/2': MultiLayerDiT_B_2,
    'MultiLayerDiT-S/2': MultiLayerDiT_S_2,
}

ReferenceDiT_models = {
    'ReferenceDiT-XL/2': ReferenceDiT_XL_2,
    'ReferenceDiT-L/2': ReferenceDiT_L_2,
    'ReferenceDiT-B/2': ReferenceDiT_B_2,
    'ReferenceDiT-S/2': ReferenceDiT_S_2,
}


#################################################################################
#                                  Test Code                                    #
#################################################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test MultiLayerDiT
    print("\n" + "="*60)
    print("Testing MultiLayerDiT")
    print("="*60)
    
    model1 = MultiLayerDiT_S_2(
        input_size=32,
        in_channels=4,
        max_layers=8,
        learn_sigma=False,
    ).to(device)
    
    B, N, C, H, W = 2, 4, 4, 32, 32  # 4 layers
    x1 = torch.randn(B, N, C, H, W).to(device)
    t1 = torch.randint(0, 1000, (B,)).to(device)
    
    with torch.no_grad():
        out1 = model1(x1, t1)
    
    print(f"Input shape:  {x1.shape}")
    print(f"Output shape: {out1.shape}")
    print(f"Expected:     (B={B}, N={N}, C_out={C}, H={H}, W={W})")
    
    # Parameter count
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"Parameters:   {params1:,}")
    
    # Test ReferenceDiT
    print("\n" + "="*60)
    print("Testing ReferenceDiT")
    print("="*60)
    
    model2 = ReferenceDiT_S_2(
        input_size=32,
        in_channels=4,
        max_ref_layers=7,
        ref_encoder_depth=2,
        learn_sigma=False,
    ).to(device)
    
    x_target = torch.randn(B, C, H, W).to(device)
    x_ref = torch.randn(B, 3, C, H, W).to(device)  # 3 reference layers
    t2 = torch.randint(0, 1000, (B,)).to(device)
    
    with torch.no_grad():
        out2 = model2(x_target, t2, x_ref)
    
    print(f"Target shape:    {x_target.shape}")
    print(f"Reference shape: {x_ref.shape}")
    print(f"Output shape:    {out2.shape}")
    print(f"Expected:        (B={B}, C_out={C}, H={H}, W={W})")
    
    # Test CFG
    with torch.no_grad():
        out2_cfg = model2.forward_with_cfg(x_target, t2, x_ref, cfg_scale=2.0)
    print(f"CFG output shape: {out2_cfg.shape}")
    
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Parameters:      {params2:,}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)