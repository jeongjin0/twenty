"""
ReferencePixArt - Single Layer Generation with Real Reference Images
실제 reference 이미지들을 입력받아 새로운 레이어 1개만 생성

사용법:
    python inference_single_layer.py \
        --checkpoint path/to/checkpoint.pth \
        --prompt "a flying bird" \
        --ref_images ref1.png ref2.png ref3.png \
        --output output.png
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.model.t5 import T5Embedder
from diffusion.model.nets.PixArt_multilayer import ReferencePixArt_XL_2


class SimpleDDIMSampler:
    """간단한 DDIM Sampler"""
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    @torch.no_grad()
    def sample(self, model, shape, y, x_ref, cfg_scale=4.5, steps=20, device='cuda'):
        """DDIM Sampling with reference"""
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
            
            null_y = torch.zeros_like(y)
            y_in = torch.cat([y, null_y], dim=0)
            x_ref_in = torch.cat([x_ref, x_ref], dim=0)
            
            noise_pred = model(x_in, t_in, y_in, x_ref_in, mask=None)
            
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


def load_image(path, size=256):
    """이미지 로드 및 전처리 (RGBA 지원)"""
    img = Image.open(path).convert('RGBA')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)  # (4, H, W), [0, 1]
    # RGB는 [-1, 1]로, Alpha는 [0, 1] 유지
    img_tensor[:3] = img_tensor[:3] * 2 - 1
    return img_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True, help="생성할 레이어 설명")
    parser.add_argument('--dataset_path', type=str, required=True, help="데이터셋 폴더 경로")
    parser.add_argument('--image_index', type=str, required=True, help="이미지 인덱스 (예: 000000117536)")
    parser.add_argument('--output', type=str, default='./generated_layer.png')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t5_path', type=str, default='/workspace/twenty/PixArt-alpha')
    parser.add_argument('--vae_path', type=str, default='/workspace/twenty/PixArt-alpha/sd-vae-ft-ema')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("ReferencePixArt - Single Layer Generation")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Image index: {args.image_index}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # ============================================
    # Load Models
    # ============================================
    print("\n[1/5] Loading models...")
    
    latent_size = args.image_size // 8

    model = ReferencePixArt_XL_2(
        input_size=latent_size,
        in_channels=5,
        max_ref_layers=7,
        ref_encoder_depth=4,
        caption_channels=4096,
        model_max_length=120,
        pred_sigma=True,
    ).to(device).eval()
    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded")
    
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()
    print(f"  ✓ VAE loaded")
    
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float16)
    print(f"  ✓ T5 loaded")
    
    # ============================================
    # Load Reference Images
    # ============================================
    print("\n[2/5] Loading reference images...")
    
    # 해당 인덱스의 모든 레이어 파일 찾기
    import glob
    pattern = os.path.join(args.dataset_path, f"{args.image_index}-layer_*.png")
    ref_paths = sorted(glob.glob(pattern))
    
    if not ref_paths:
        raise FileNotFoundError(f"No files found for index {args.image_index} in {args.dataset_path}")
    
    ref_images = []
    for path in ref_paths:
        img = load_image(path, args.image_size)
        ref_images.append(img)
        print(f"  ✓ Loaded: {os.path.basename(path)}")
    
    ref_images = torch.stack(ref_images, dim=0).to(device)  # (N_ref, 3, H, W)
    print(f"  Reference tensor: {ref_images.shape} ({len(ref_paths)} layers)")
    
    # ============================================
    # Encode References with VAE
    # ============================================
    print("\n[3/5] Encoding references with VAE...")
    
    with torch.no_grad():
        # RGB와 Alpha 분리
        rgb = ref_images[:, :3, :, :]   # (N_ref, 3, H, W)
        alpha = ref_images[:, 3:4, :, :]  # (N_ref, 1, H, W)
        
        # RGB: VAE 인코딩
        z_rgb = vae.encode(rgb).latent_dist.mode() * 0.18215  # (N_ref, 4, h, w)
        
        # Alpha: 단순 다운샘플링 (VAE 거치지 않음)
        h, w = z_rgb.shape[-2:]
        alpha_down = F.interpolate(alpha, size=(h, w), mode='bilinear', align_corners=False)  # (N_ref, 1, h, w)
        
        # RGB latent (4ch) + Alpha (1ch) = 5채널
        ref_latents = torch.cat([z_rgb, alpha_down], dim=1)  # (N_ref, 5, h, w)
    
    ref_latents = ref_latents.unsqueeze(0)  # (1, N_ref, 5, h, w)
    print(f"  Reference latents: {ref_latents.shape}")
    
    # ============================================
    # Encode Text
    # ============================================
    while True:
        print("\n[4/5] Encoding text...")
        args.prompt = input("Enter prompt (or 'exit' to quit): ")
        
        caption_embs, _ = t5.get_text_embeddings([args.prompt])
        y = caption_embs.float()[:, None]  # (1, 1, L, 4096)
        print(f"  Text embedding: {y.shape}")
        
        # ============================================
        # Generate Target Layer
        # ============================================
        print("\n[5/5] Generating target layer...")
        
        sampler = SimpleDDIMSampler()
        
        z = sampler.sample(
            model=model,
            shape=(1, 5, latent_size, latent_size),
            y=y,
            x_ref=ref_latents,
            cfg_scale=args.cfg_scale,
            steps=args.steps,
            device=device,
        )
        
        # ============================================
        # Decode & Save
        # ============================================
        print("\nDecoding and saving...")
        
        with torch.no_grad():
            # z는 5채널 (4 RGB latent + 1 Alpha)
            z_rgb = z[:, :4, :, :]  # (1, 4, h, w)
            z_alpha = z[:, 4:5, :, :]  # (1, 1, h, w)
            
            # RGB 디코딩
            generated_rgb = vae.decode(z_rgb / 0.18215).sample  # (1, 3, H, W)
            
            # Alpha 업샘플링
            H, W = generated_rgb.shape[-2:]
            generated_alpha = F.interpolate(z_alpha, size=(H, W), mode='bilinear', align_corners=False)
            
            # RGBA 합치기
            generated_img = torch.cat([generated_rgb, generated_alpha], dim=1)  # (1, 4, H, W)        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        
        # Save generated layer
        save_image(generated_img, args.output, normalize=True, value_range=(-1, 1))
        print(f"  ✓ Saved generated layer: {args.output}")
        
        # Save comparison (refs + generated)
        base_name = os.path.splitext(args.output)[0]
        
        # All images grid: [ref1, ref2, ..., generated]
        # 비교를 위해 RGB만 사용 (또는 RGBA 모두 저장)
        ref_rgb = ref_images[:, :3, :, :]  # (N_ref, 3, H, W)
        gen_rgb = generated_img[:, :3, :, :]  # (1, 3, H, W)
        all_images = torch.cat([ref_rgb, gen_rgb], dim=0)
        
        comparison_path = f"{base_name}_comparison.png"
        save_image(all_images, comparison_path, nrow=len(ref_paths)+1, normalize=True, value_range=(-1, 1))
        print(f"  ✓ Saved comparison: {comparison_path}")
        
        print("\n" + "="*60)
        print("Done!")
        print("="*60)


if __name__ == '__main__':
    main()