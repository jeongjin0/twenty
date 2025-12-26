# ReferencePixArt with Cross-Attention Reference Injection
# 
# 두 가지 Reference 주입 방식:
# 1. AdaLN 방식 (기존): ref_emb를 timestep에 더해서 adaLN으로 주입
# 2. Cross-Attention 방식 (새로운): ref_tokens를 text와 concat해서 cross-attention
#
# Cross-Attention 장점:
# - 더 풍부한 공간적 상호작용
# - Reference의 세부 정보 더 잘 전달
# - Attention map으로 어디를 참조하는지 시각화 가능

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp
from typing import Optional, List, Tuple

# Import from PixArt_blocks for consistency and bug-free implementations
from diffusion.model.nets.PixArt_blocks import (
    t2i_modulate,
    CaptionEmbedder,
    WindowAttention,
    MultiHeadCrossAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    PixArtBlock
)


#################################################################################
#                              Utility Functions                                #
#################################################################################

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
#                    Reference Encoder (Token-based for Cross-Attention)        #
#################################################################################

class ReferenceTokenEncoder(nn.Module):
    """
    Reference 레이어들을 토큰 시퀀스로 인코딩.
    Cross-attention에서 사용하기 위해 (B, N_tokens, D) 형태로 출력.
    
    vs ReferenceEncoder (기존):
    - 기존: 모든 reference를 하나의 벡터로 압축 (B, D)
    - 새로운: 토큰 시퀀스 유지 (B, N_ref * T_compressed, D)
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
        compression_ratio=4,  # 공간 압축 비율 (T → T/ratio)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_layers = max_layers
        self.compression_ratio = compression_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches
        
        # Compressed token count per layer
        self.tokens_per_layer = num_patches // (compression_ratio ** 2)

        # Positional embeddings
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        self.layer_embed = nn.Parameter(torch.zeros(1, max_layers, 1, hidden_size))

        # Spatial compression (learnable pooling)
        self.spatial_compress = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=compression_ratio, stride=compression_ratio),
            nn.GELU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
        )

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

        # Output projection to match text embedding dimension
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer type embedding (구분: text vs reference)
        self.ref_type_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        nn.init.normal_(self.layer_embed, std=0.02)
        nn.init.normal_(self.ref_type_embed, std=0.02)
        
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, ref_layers):
        """
        ref_layers: (B, N_ref, C, H, W)
        Returns: (B, N_ref * T_compressed, D) - token sequence for cross-attention
        """
        B, N_ref, C, H, W = ref_layers.shape

        # Patch embed
        x = ref_layers.reshape(B * N_ref, C, H, W)
        x = self.patch_embed(x)  # (B*N_ref, T, D)
        
        T, D = x.shape[1], x.shape[2]
        h = w = int(T ** 0.5)
        
        # Add positional embeddings before compression
        x = x + self.pos_embed
        
        # Reshape for spatial compression
        x = x.reshape(B * N_ref, h, w, D).permute(0, 3, 1, 2)  # (B*N_ref, D, h, w)
        
        # Spatial compression
        x = self.spatial_compress(x)  # (B*N_ref, D, h', w')
        h_comp, w_comp = x.shape[2], x.shape[3]
        T_comp = h_comp * w_comp
        
        x = x.permute(0, 2, 3, 1).reshape(B * N_ref, T_comp, D)  # (B*N_ref, T_comp, D)
        
        # Reshape to (B, N_ref, T_comp, D)
        x = x.reshape(B, N_ref, T_comp, D)
        
        # Add layer embeddings
        x = x + self.layer_embed[:, :N_ref, :, :]
        
        # Flatten: (B, N_ref * T_comp, D)
        x = x.reshape(B, N_ref * T_comp, D)
        
        # Add reference type embedding
        x = x + self.ref_type_embed
        
        # Transformer encode
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x))
            x = x + block['mlp'](block['norm2'](x))
        
        # Output projection
        x = self.output_proj(x)
        
        return x  # (B, N_ref * T_comp, D)


#################################################################################
#         ReferencePixArt with Cross-Attention (CrossAttn Version)              #
#################################################################################

class ReferencePixArtCrossAttn(nn.Module):
    """
    Cross-Attention 방식의 Reference 주입.
    
    기존 AdaLN 방식과 비교:
    - AdaLN: ref_emb (B, D)를 timestep에 더해서 전체 네트워크에 broadcast
    - CrossAttn: ref_tokens (B, N_tokens, D)를 text tokens와 concat하여 cross-attention
    
    장점:
    - 공간적 세부 정보 더 잘 전달
    - Reference의 어떤 부분을 참조하는지 attention으로 학습
    - Text와 Reference 정보의 명시적 분리
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
        ref_compression_ratio=4,
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
        self.max_ref_layers = max_ref_layers

        # Target processing
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

        # Reference Token Encoder (새로운!)
        self.ref_encoder = ReferenceTokenEncoder(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_heads=num_heads,
            depth=ref_encoder_depth,
            mlp_ratio=mlp_ratio,
            max_layers=max_ref_layers,
            compression_ratio=ref_compression_ratio,
        )

        # Transformer blocks
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
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False

        self.initialize_weights()

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

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x_target, timestep, y, x_ref, mask=None, **kwargs):
        """
        x_target: (B, C, H, W) - 노이즈 추가된 타겟
        timestep: (B,) - diffusion timesteps
        y: (B, 1, L, caption_channels) - text embeddings
        x_ref: (B, N_ref, C, H, W) - 참조 레이어들 (clean)
        """
        dtype = next(self.parameters()).dtype
        x_target = x_target.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)
        x_ref = x_ref.to(dtype)

        B = x_target.shape[0]
        N_ref = x_ref.shape[1]

        # 1. Target embedding
        x = self.x_embedder(x_target) + self.pos_embed  # (B, T, D)

        # 2. Timestep embedding (no reference added here!)
        t = self.t_embedder(timestep)
        t0 = self.t_block(t)

        # 3. Text embedding
        y = self.y_embedder(y, self.training)  # (B, 1, L, D)
        D = x.shape[-1]
        y = y.squeeze(1)  # (B, L, D)

        # 4. Reference token encoding (핵심!)
        ref_tokens = self.ref_encoder(x_ref)  # (B, N_ref * T_comp, D)

        # 5. Concatenate text + reference tokens for cross-attention
        # conditioning = [text_tokens, ref_tokens]
        # Shape: (B, L + N_ref*T_comp, D)
        cond_tokens = torch.cat([y, ref_tokens], dim=1)

        # 6. Transformer blocks with combined cross-attention
        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, cond_tokens, t0, None, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, cond_tokens, t0)

        # 7. Final layer
        x = self.final_layer(x, t)

        # 8. Unpatchify
        x = self.unpatchify(x)

        return x

    def forward_without_ref(self, x_target, timestep, y, mask=None):
        """Reference 없이 forward (CFG용)"""
        dtype = next(self.parameters()).dtype
        x_target = x_target.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        B = x_target.shape[0]
        x = self.x_embedder(x_target) + self.pos_embed

        t = self.t_embedder(timestep)
        t0 = self.t_block(t)

        y = self.y_embedder(y, self.training)
        D = x.shape[-1]
        y = y.squeeze(1)  # (B, L, D)

        # Reference 없이 text만 사용
        cond_tokens = y

        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, cond_tokens, t0, None, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, cond_tokens, t0)

        x = self.final_layer(x, t)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(self, x_target, timestep, y, x_ref, cfg_scale, mask=None, **kwargs):
        """Classifier-free guidance"""
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

def ReferencePixArtCrossAttn_XL_2(**kwargs):
    return ReferencePixArtCrossAttn(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def ReferencePixArtCrossAttn_L_2(**kwargs):
    return ReferencePixArtCrossAttn(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def ReferencePixArtCrossAttn_B_2(**kwargs):
    return ReferencePixArtCrossAttn(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


#################################################################################
#                                  Test Code                                    #
#################################################################################

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Test parameters
    B = 2
    N_ref = 3
    C = 4
    H = W = 32  # latent size for 256px
    caption_channels = 4096
    L = 120

    print("\n" + "="*60)
    print("Testing ReferencePixArtCrossAttn")
    print("="*60)

    model = ReferencePixArtCrossAttn_B_2(
        input_size=H,
        in_channels=C,
        max_ref_layers=7,
        ref_encoder_depth=2,
        ref_compression_ratio=2,  # 압축 비율
        caption_channels=caption_channels,
        model_max_length=L,
        pred_sigma=False,
    ).to(device)

    x_target = torch.randn(B, C, H, W).to(device)
    x_ref = torch.randn(B, N_ref, C, H, W).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    y = torch.randn(B, 1, L, caption_channels).to(device)

    print(f"Target shape:     {x_target.shape}")
    print(f"Reference shape:  {x_ref.shape}")
    print(f"Text shape:       {y.shape}")

    # Test reference token encoder
    ref_tokens = model.ref_encoder(x_ref)
    print(f"Ref tokens shape: {ref_tokens.shape}")
    print(f"  (N_ref={N_ref}, T_compressed={ref_tokens.shape[1]//N_ref})")

    with torch.no_grad():
        out = model(x_target, t, y, x_ref)

    print(f"Output shape:     {out.shape}")
    print(f"Expected:         (B={B}, C_out={C}, H={H}, W={W})")

    # Test without ref
    with torch.no_grad():
        out_no_ref = model.forward_without_ref(x_target, t, y)
    print(f"No-ref output:    {out_no_ref.shape}")

    # Test CFG
    with torch.no_grad():
        out_cfg = model.forward_with_cfg(x_target, t, y, x_ref, cfg_scale=4.5)
    print(f"CFG output:       {out_cfg.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:       {params:,}")

    # Memory estimate
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x_target, t, y, x_ref)
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak memory:      {mem:.2f} GB (inference)")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)