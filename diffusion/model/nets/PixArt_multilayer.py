# Multi-Layer Text-to-Image Architectures based on PixArt
# Architecture 1: MultiLayerPixArt - All layers simultaneous generation with text conditioning
# Architecture 2: ReferencePixArt - Single layer generation with reference + text conditioning
#
# References:
# PixArt-α: https://github.com/PixArt-alpha/PixArt-alpha
# DiT: https://github.com/facebookresearch/DiT

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp
from typing import Optional, List, Tuple


#################################################################################
#                              Utility Functions                                #
#################################################################################

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, lewei_scale=1.0, base_size=16):
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0]/base_size) / lewei_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1]/base_size) / lewei_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


#################################################################################
#                              Core Components                                  #
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


class CaptionEmbedder(nn.Module):
    """
    Embeds text captions into vector representations.
    Also handles unconditional embedding for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU, token_num=120):
        super().__init__()
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            act_layer(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """Drops captions to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding.unsqueeze(0).unsqueeze(0), caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross attention for text conditioning."""
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        """
        x: (B, N, C) - image tokens
        cond: (1, L_total, C) or (B, L, C) - text tokens (packed or batched)
        mask: optional attention mask
        """
        B, N, C = x.shape

        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Handle packed text tokens
        if cond.shape[0] == 1 and B > 1:
            # Packed format: (1, L_total, C)
            kv = self.kv_linear(cond).reshape(1, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # (1, num_heads, L_total, head_dim)
            k = k.expand(B, -1, -1, -1)
            v = v.expand(B, -1, -1, -1)
        else:
            # Batched format: (B, L, C)
            kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttention(nn.Module):
    """Self-attention with optional window attention."""
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 input_size=None, use_rel_pos=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_rel_pos = use_rel_pos
        if use_rel_pos and input_size is not None:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class T2IFinalLayer(nn.Module):
    """Final layer for text-to-image models."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t.reshape(-1, 2, x.shape[-1])).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                              PixArt Block                                     #
#################################################################################

class PixArtBlock(nn.Module):
    """A PixArt block with adaptive layer norm (adaLN-single) conditioning."""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., 
                 window_size=0, input_size=None, use_rel_pos=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            input_size=input_size if window_size == 0 else (window_size, window_size),
            use_rel_pos=use_rel_pos, **block_kwargs
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), 
                       act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        
        x = x + self.drop_path(gate_msa * self.attn(
            t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        ).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        ))
        return x


#################################################################################
#       Architecture 1: MultiLayerPixArt - Simultaneous Layer Generation       #
#################################################################################

class MultiLayerPixArtBlock(nn.Module):
    """
    PixArt block adapted for multi-layer generation.
    Self-attention operates across all layers (layer-wise attention).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0.,
                 input_size=None, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Self-attention (will handle multi-layer)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        
        # Cross-attention for text
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu, drop=0)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, num_layers=None, num_patches=None, **kwargs):
        """
        x: (B, N*T, C) - flattened multi-layer tokens
        y: text conditioning
        t: timestep modulation (B, 6, C)
        num_layers: N
        num_patches: T
        """
        B, NT, C = x.shape
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        
        # Self-attention across all layers
        x_norm = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self._multi_layer_attention(x_norm, num_layers, num_patches)
        x = x + self.drop_path(gate_msa * attn_out)
        
        # Cross-attention to text
        x = x + self.cross_attn(x, y, mask)
        
        # MLP
        x = x + self.drop_path(gate_mlp * self.mlp(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        ))
        
        return x

    def _multi_layer_attention(self, x, num_layers, num_patches):
        """Full attention across all layers and patches."""
        B, NT, C = x.shape
        
        qkv = self.qkv(x).reshape(B, NT, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, NT, C)
        x = self.attn_proj(x)
        return x


class MultiLayerPixArt(nn.Module):
    """
    Multi-Layer PixArt for text-to-multi-layer-image generation.
    
    Input: 
        - x: (B, N, C, H, W) - N layers of images
        - timestep: (B,)
        - y: (B, 1, L, caption_channels) - text embeddings
    Output: (B, N, C_out, H, W)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.,
        caption_channels=4096,
        lewei_scale=1.0,
        model_max_length=120,
        max_layers=16,
        **kwargs
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_layers = max_layers
        self.lewei_scale = lewei_scale
        self.base_size = input_size // patch_size

        # Patch embedding (shared)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        num_patches = self.x_embedder.num_patches
        self.num_patches = num_patches
        
        # Spatial positional embedding (fixed)
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        
        # Layer positional embedding (learnable)
        self.layer_embed = nn.Parameter(torch.zeros(1, max_layers, 1, hidden_size))

        # Timestep modulation
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Caption embedder
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels, 
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob, 
            act_layer=approx_gelu,
            token_num=model_max_length
        )

        # Transformer blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            MultiLayerPixArtBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                drop_path=drop_path_rates[i],
                input_size=(input_size // patch_size, input_size // patch_size)
            )
            for i in range(depth)
        ])

        # Final layer
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory savings."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Spatial pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.num_patches ** 0.5),
            lewei_scale=self.lewei_scale, 
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Layer embed
        nn.init.normal_(self.layer_embed, std=0.02)

        # Patch embed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Timestep embedding
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Caption embedding
        nn.init.normal_(self.y_embedder.y_proj[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj[2].weight, std=0.02)

        # Zero-out cross attention output
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, num_layers):
        """
        x: (B, N*T, patch_size**2 * C)
        imgs: (B, N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)

        B = x.shape[0]
        T = self.num_patches

        x = x.reshape(B, num_layers, T, -1)
        x = x.reshape(B, num_layers, h, w, p, p, c)
        x = torch.einsum('bnhwpqc->bncphwq', x)
        imgs = x.reshape(B, num_layers, c, h * p, w * p)
        return imgs

    def forward(self, x, timestep, y, mask=None, **kwargs):
        """
        x: (B, N, C, H, W) - N layers of images
        timestep: (B,) tensor of diffusion timesteps
        y: (B, 1, L, caption_channels) tensor of text embeddings
        mask: optional text mask
        """
        B, N, C, H, W = x.shape
        assert N <= self.max_layers

        # Cast to model dtype
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # 1. Patch embed each layer
        x = x.reshape(B * N, C, H, W)
        x = self.x_embedder(x)  # (B*N, T, D)
        
        T, D = x.shape[1], x.shape[2]
        x = x.reshape(B, N, T, D)

        # 2. Add positional embeddings
        x = x + self.pos_embed.unsqueeze(1)  # spatial
        x = x + self.layer_embed[:, :N, :, :]  # layer

        # 3. Flatten: (B, N, T, D) -> (B, N*T, D)
        x = x.reshape(B, N * T, D)

        # 4. Timestep embedding
        t = self.t_embedder(timestep)
        t0 = self.t_block(t)

        # 5. Text embedding
        y = self.y_embedder(y, self.training)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, D)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, D)

        # 6. Transformer blocks
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, y, t0, None, N, T, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, y, t0, mask=None, num_layers=N, num_patches=T)

        # 7. Final layer
        t_final = self.t_embedder(timestep)
        t_final = self.t_block(t_final)[:, :2*D].reshape(B, 2, D)
        x = self.final_layer(x, t_final)

        # 8. Unpatchify
        x = self.unpatchify(x, N)

        return x

    def forward_with_cfg(self, x, timestep, y, cfg_scale, mask=None, **kwargs):
        """Forward with classifier-free guidance."""
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask)
        
        eps, rest = model_out[:, :, :self.in_channels], model_out[:, :, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)

    def load_pretrained_pixart(self, state_dict, strict=False):
        """
        Load pretrained PixArt weights.
        
        Compatible weights:
        - x_embedder, t_embedder, t_block: fully compatible
        - pos_embed: compatible
        - y_embedder: compatible
        - blocks: partially compatible (cross_attn, mlp, norms)
        - final_layer: compatible
        
        New weights (random init):
        - layer_embed
        - blocks.*.qkv, blocks.*.attn_proj (replaced WindowAttention)
        """
        model_dict = self.state_dict()
        
        # Map old attention weights to new format
        mapped_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                mapped_dict[k] = v
            elif 'blocks' in k and 'attn.qkv' in k:
                # Map WindowAttention.qkv to our qkv
                new_k = k.replace('attn.qkv', 'qkv')
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    mapped_dict[new_k] = v
            elif 'blocks' in k and 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn_proj')
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    mapped_dict[new_k] = v
        
        model_dict.update(mapped_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"Loaded {len(mapped_dict)}/{len(state_dict)} pretrained weights")
        return list(set(state_dict.keys()) - set(mapped_dict.keys()))


#################################################################################
#       Architecture 2: ReferencePixArt - Reference-based Generation            #
#################################################################################

class ReferenceEncoder(nn.Module):
    """
    Encodes multiple reference layers into conditioning embeddings.
    Uses a lightweight transformer to aggregate reference information.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        num_heads=16,
        depth=4,
        mlp_ratio=4.0,
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
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        self.layer_embed = nn.Parameter(torch.zeros(1, max_layers, 1, hidden_size))

        # Pool token
        self.pool_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Transformer encoder
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_size, eps=1e-6),
                'attn': WindowAttention(hidden_size, num_heads=num_heads, qkv_bias=True),
                'norm2': nn.LayerNorm(hidden_size, eps=1e-6),
                'mlp': Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio),
                          act_layer=approx_gelu, drop=0),
            })
            for _ in range(depth)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        nn.init.normal_(self.layer_embed, std=0.02)
        nn.init.normal_(self.pool_token, std=0.02)
        
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, ref_layers):
        """
        ref_layers: (B, N_ref, C, H, W)
        Returns: (B, D) aggregated reference embedding
        """
        B, N_ref, C, H, W = ref_layers.shape

        # Patch embed
        x = ref_layers.reshape(B * N_ref, C, H, W)
        x = self.patch_embed(x)
        
        T, D = x.shape[1], x.shape[2]
        x = x.reshape(B, N_ref, T, D)

        # Add positional embeddings
        x = x + self.pos_embed.unsqueeze(1)
        x = x + self.layer_embed[:, :N_ref, :, :]

        # Flatten
        x = x.reshape(B, N_ref * T, D)

        # Prepend pool token
        pool_tokens = self.pool_token.expand(B, -1, -1)
        x = torch.cat([pool_tokens, x], dim=1)

        # Encode
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x))
            x = x + block['mlp'](block['norm2'](x))

        # Extract pool token
        pooled = x[:, 0]
        output = self.output_proj(pooled)

        return output


class ReferencePixArt(nn.Module):
    """
    Reference-based PixArt for single layer generation with text + reference conditioning.
    
    Input:
        - x_target: (B, C, H, W) - target layer to generate
        - timestep: (B,)
        - y: (B, 1, L, caption_channels) - text embeddings
        - x_ref: (B, N_ref, C, H, W) - reference layers
    Output: (B, C_out, H, W)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.,
        caption_channels=4096,
        lewei_scale=1.0,
        model_max_length=120,
        max_ref_layers=15,
        ref_encoder_depth=4,
        **kwargs
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.lewei_scale = lewei_scale
        self.base_size = input_size // patch_size

        # Target processing (standard PixArt)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        num_patches = self.x_embedder.num_patches
        self.num_patches = num_patches
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        # Timestep modulation
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Caption embedder
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length
        )

        # Reference encoder
        self.ref_encoder = ReferenceEncoder(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=ref_encoder_depth,
            mlp_ratio=mlp_ratio,
            max_layers=max_ref_layers,
        )

        # Reference conditioning projection
        self.ref_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Transformer blocks (standard PixArt blocks)
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            PixArtBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                drop_path=drop_path_rates[i],
                input_size=(input_size // patch_size, input_size // patch_size)
            )
            for i in range(depth)
        ])

        # Final layer
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory savings."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            lewei_scale=self.lewei_scale,
            base_size=self.base_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_proj[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x_target, timestep, y, x_ref, mask=None, **kwargs):
        dtype = next(self.parameters()).dtype
        x_target = x_target.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        x_ref = x_ref.to(dtype)

        B = x_target.shape[0]

        # 1. Encode target
        x = self.x_embedder(x_target) + self.pos_embed

        # 2. Encode reference
        ref_emb = self.ref_encoder(x_ref)
        ref_emb = self.ref_proj(ref_emb)

        # 3. Timestep embedding + reference conditioning
        t = self.t_embedder(timestep)
        t = t + ref_emb
        t0 = self.t_block(t)

        # 4. Text embedding
        y = self.y_embedder(y, self.training)
        D = x.shape[-1]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, D)
        else:
            y = y.squeeze(1).view(1, -1, D)

        # 5. Transformer blocks (수정됨)
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, y, t0, None, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, y, t0)

        # 6. Final layer (squeeze 제거)
        t_final = t0[:, :2*D].reshape(B, 2, D)
        x = self.final_layer(x, t_final)

        # 7. Unpatchify
        x = self.unpatchify(x)

        return x

    def forward_without_ref(self, x_target, timestep, y, mask=None):
        """Forward without reference (for CFG)."""
        dtype = next(self.parameters()).dtype
        x_target = x_target.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        B = x_target.shape[0]
        x = self.x_embedder(x_target) + self.pos_embed

        # Zero reference
        ref_emb = torch.zeros(B, self.hidden_size, device=x_target.device, dtype=dtype)
        ref_emb = self.ref_proj(ref_emb)

        t = self.t_embedder(timestep) + ref_emb
        t0 = self.t_block(t)

        y = self.y_embedder(y, self.training)
        D = x.shape[-1]
        if mask is not None:
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, D)
        else:
            y = y.squeeze(1).view(1, -1, D)

        # Transformer blocks (수정됨)
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, y, t0, None, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, y, t0)

        # Final layer (squeeze 제거)
        t_final = t0[:, :2*D].reshape(B, 2, D)
        x = self.final_layer(x, t_final)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(self, x_target, timestep, y, x_ref, cfg_scale, mask=None, **kwargs):
        """Forward with classifier-free guidance on both text and reference."""
        cond_out = self.forward(x_target, timestep, y, x_ref, mask)
        uncond_out = self.forward_without_ref(x_target, timestep, y, mask)

        if self.pred_sigma:
            eps_cond, rest = cond_out[:, :self.in_channels], cond_out[:, self.in_channels:]
            eps_uncond, _ = uncond_out[:, :self.in_channels], uncond_out[:, self.in_channels:]
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            return torch.cat([eps, rest], dim=1)
        else:
            return uncond_out + cfg_scale * (cond_out - uncond_out)

    def load_pretrained_pixart(self, state_dict, strict=False):
        """Load pretrained PixArt weights."""
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=strict)
        print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} pretrained weights")
        return list(set(state_dict.keys()) - set(pretrained_dict.keys()))


#################################################################################
#                              Model Configurations                             #
#################################################################################

def MultiLayerPixArt_XL_2(**kwargs):
    return MultiLayerPixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def MultiLayerPixArt_L_2(**kwargs):
    return MultiLayerPixArt(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def MultiLayerPixArt_B_2(**kwargs):
    return MultiLayerPixArt(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def ReferencePixArt_XL_2(**kwargs):
    return ReferencePixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def ReferencePixArt_L_2(**kwargs):
    return ReferencePixArt(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def ReferencePixArt_B_2(**kwargs):
    return ReferencePixArt(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


MultiLayerPixArt_models = {
    'MultiLayerPixArt-XL/2': MultiLayerPixArt_XL_2,
    'MultiLayerPixArt-L/2': MultiLayerPixArt_L_2,
    'MultiLayerPixArt-B/2': MultiLayerPixArt_B_2,
}

ReferencePixArt_models = {
    'ReferencePixArt-XL/2': ReferencePixArt_XL_2,
    'ReferencePixArt-L/2': ReferencePixArt_L_2,
    'ReferencePixArt-B/2': ReferencePixArt_B_2,
}


#################################################################################
#                                  Test Code                                    #
#################################################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test parameters
    B = 2
    N = 4  # num layers
    C = 4  # RGBA
    H = W = 64  # image size (will be 64x64, latent of 512x512)
    caption_channels = 4096  # T5-XXL dimension
    L = 120  # max text length

    # Test MultiLayerPixArt
    print("\n" + "="*60)
    print("Testing MultiLayerPixArt")
    print("="*60)

    model1 = MultiLayerPixArt_B_2(
        input_size=H,
        in_channels=C,
        max_layers=8,
        caption_channels=caption_channels,
        model_max_length=L,
        pred_sigma=False,
    ).to(device)

    x1 = torch.randn(B, N, C, H, W).to(device)
    t1 = torch.randint(0, 1000, (B,)).to(device)
    y1 = torch.randn(B, 1, L, caption_channels).to(device)

    with torch.no_grad():
        out1 = model1(x1, t1, y1)

    print(f"Input shape:   {x1.shape}")
    print(f"Text shape:    {y1.shape}")
    print(f"Output shape:  {out1.shape}")
    print(f"Expected:      (B={B}, N={N}, C_out={C}, H={H}, W={W})")

    params1 = sum(p.numel() for p in model1.parameters())
    print(f"Parameters:    {params1:,}")

    # Test ReferencePixArt
    print("\n" + "="*60)
    print("Testing ReferencePixArt")
    print("="*60)

    model2 = ReferencePixArt_B_2(
        input_size=H,
        in_channels=C,
        max_ref_layers=7,
        ref_encoder_depth=2,
        caption_channels=caption_channels,
        model_max_length=L,
        pred_sigma=False,
    ).to(device)

    x_target = torch.randn(B, C, H, W).to(device)
    x_ref = torch.randn(B, 3, C, H, W).to(device)
    t2 = torch.randint(0, 1000, (B,)).to(device)
    y2 = torch.randn(B, 1, L, caption_channels).to(device)

    with torch.no_grad():
        out2 = model2(x_target, t2, y2, x_ref)

    print(f"Target shape:     {x_target.shape}")
    print(f"Reference shape:  {x_ref.shape}")
    print(f"Text shape:       {y2.shape}")
    print(f"Output shape:     {out2.shape}")
    print(f"Expected:         (B={B}, C_out={C}, H={H}, W={W})")

    # Test CFG
    with torch.no_grad():
        out2_cfg = model2.forward_with_cfg(x_target, t2, y2, x_ref, cfg_scale=4.5)
    print(f"CFG output shape: {out2_cfg.shape}")

    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Parameters:       {params2:,}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)