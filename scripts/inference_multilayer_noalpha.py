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
from diffusion.sampler import SimpleDDIMSampler


def load_image(path, size=256):
    """이미지 로드 및 전처리 (RGBA 지원)"""
    img = Image.open(path).convert('RGBA')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),  # RGBA 모두 [-1, 1]
    ])
    return transform(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prompt', type=str, help="생성할 레이어 설명")
    parser.add_argument('--dataset_path', type=str, required=True, help="데이터셋 폴더 경로")
    parser.add_argument('--image_index', type=str, required=True, help="이미지 인덱스 (예: 000000117536)")
    parser.add_argument('--output', type=str, default='./generated_layer.png')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--cfg_scale', type=float, default=4.5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t5_path', type=str, default='/workspace/twenty/PixArt-alpha')
    parser.add_argument('--vae_path', type=str, default='/workspace/twenty/PixArt-alpha/sd-vae-ft-ema')
    parser.add_argument("--use_ref", action="store_true")
    parser.add_argument("--no_use_ref", action="store_false", dest="use_ref")
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

    # model = ReferencePixArt_XL_2(
    #     input_size=latent_size,
    #     in_channels=4,
    #     max_ref_layers=7,
    #     ref_encoder_depth=4,
    #     caption_channels=4096,
    #     model_max_length=120,
    #     pred_sigma=True,
    #     use_ref=args.use_ref,
    # ).to(device).eval()

    from diffusion.model.nets.PixArt_reference_crossattn import ReferencePixArtCrossAttn_XL_2


    model = ReferencePixArtCrossAttn_XL_2(
        input_size=latent_size,
        in_channels=4,
        max_ref_layers=7,
        ref_encoder_depth=4,
        ref_compression_ratio=4,
        caption_channels=4096,
        model_max_length=120,
        pred_sigma=True,
    ).to(device).eval()

    print("args use ref:", args.use_ref)


    # from diffusion.model.nets.PixArt import PixArt_XL_2

    # model = PixArt_XL_2(
    #     input_size=latent_size,
    #     in_channels=4,
    #     caption_channels=4096,
    #     model_max_length=120,
    #     pred_sigma=True,
    # ).to(device).eval()

    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    missing_keys = model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ Model loaded")
    print(f"    Missing keys: {missing_keys.missing_keys}")
    
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
        ref_latents = z_rgb    

    ref_latents = ref_latents.unsqueeze(0)  # (1, N_ref, 4, h, w)
    print(f"  Reference latents: {ref_latents.shape}")
    print(f"ref_latents mean: {ref_latents.mean():.4f}, std: {ref_latents.std():.4f}")
    print(f"ref_latents[:, :, :4] (RGB) std: {ref_latents[:, :, :4].std():.4f}")  # RGB
    print(f"ref_latents[:, :, 4:] (Alpha) std: {ref_latents[:, :, 4:].std():.4f}")  # Alpha


    with torch.no_grad():
        ref_z_rgb = ref_latents[0, 0, :4]  # 첫번째 ref, RGB만 (4, 8, 8)
        ref_z_rgb = ref_z_rgb.unsqueeze(0)  # (1, 4, 8, 8)
        ref_decoded = vae.decode(ref_z_rgb / 0.18215).sample
        print(f"Ref decoded mean: {ref_decoded.mean():.4f}, std: {ref_decoded.std():.4f}")
        save_image(ref_decoded, "ref_decoded_test.png", normalize=True, value_range=(-1, 1))


    # ============================================
    # Encode Text
    # ============================================
    while True:
        print("\n[4/5] Encoding text...")
        args.prompt = input("Enter prompt (or 'exit' to quit): ")
        
        caption_embs, mask = t5.get_text_embeddings([args.prompt])
        y = caption_embs.float()[:, None]  # (1, 1, L, 4096)

        print(f"  Text embedding: {y.shape}")
        
        # ============================================
        # Generate Target Layer
        # ============================================
        print("\n[5/5] Generating target layer...")
        
        sampler = SimpleDDIMSampler()
        
        z = sampler.sample(
            model=model,
            shape=(1, 4, latent_size, latent_size),
            y=y,
            y_mask=mask,
            x_ref=ref_latents,
            cfg_scale=args.cfg_scale,
            steps=args.steps,
            device=device,
        )


        # ============================================
        # Decode & Save
        # ============================================

        with torch.no_grad():
            generated_rgb = vae.decode(z / 0.18215).sample
            
            # RGBA 합치기
            generated_img = generated_rgb
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