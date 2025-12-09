"""
MuLan Multi-Layer Dataset with Text Captions
- 각 이미지는 여러 개의 RGBA 레이어로 구성
- 각 레이어별 caption (blip2, llava) 지원
- Reference-based architecture용
"""

import os
import re
import csv
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F


class MuLanDataset(Dataset):
    """
    MuLan Multi-Layer RGBA Dataset with Text Captions
    
    Returns:
        layers: (N_layers, 4, H, W) - RGBA layers
        captions: List[str] - caption for each layer
        num_layers: int - actual number of layers (before padding)
        image_id: str - image identifier
    """
    
    def __init__(
        self,
        data_roots: List[str],
        resolution: int = 512,
        max_layers: int = 8,
        min_layers: int = 2,
        caption_type: str = "blip2",  # "blip2" or "llava"
    ):
        """
        Args:
            data_roots: List of directories containing layer images
            resolution: Target resolution (square)
            max_layers: Maximum number of layers to load (pad if fewer)
            min_layers: Minimum number of layers required
            caption_type: Which caption to use ("blip2" or "llava")
        """
        self.resolution = resolution
        self.max_layers = max_layers
        self.min_layers = min_layers
        self.caption_type = caption_type
        
        # Load metadata and collect image groups
        self.image_groups = self._load_from_csv(data_roots)
        self.image_ids = list(self.image_groups.keys())
        
        print(f"[MuLanDataset] Found {len(self.image_ids)} images")
        print(f"[MuLanDataset] Resolution: {resolution}, Max layers: {max_layers}")
        print(f"[MuLanDataset] Caption type: {caption_type}")
        
        # Transform
        self.transform = T.Compose([
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # [0, 255] -> [0, 1]
        ])
    
    def _load_from_csv(self, data_roots: List[str]) -> Dict[str, Dict]:
        """
        Load metadata from CSV files and group by image_id
        
        Expected CSV format:
            paths,blip2,llava
            data/mulan_laion/000635517-layer_0.png,caption1,caption2
        """
        image_groups = defaultdict(lambda: {"layers": [], "captions": []})
        pattern = re.compile(r'(\d+)-layer_(\d+)\.png$')
        
        for root in data_roots:
            root = Path(root)
            csv_path = root / "meta_data.csv"
            
            if not csv_path.exists():
                print(f"[Warning] CSV not found: {csv_path}")
                continue
            
            print(f"[MuLanDataset] Loading CSV: {csv_path}")
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    file_path = row['paths']
                    blip2_caption = row.get('blip2', '')
                    llava_caption = row.get('llava', '')
                    
                    # Extract image_id and layer_idx from path
                    match = pattern.search(file_path)
                    if not match:
                        continue
                    
                    image_id_num = match.group(1)
                    layer_idx = int(match.group(2))
                    
                    # Full path construction
                    # CSV has relative path like "data/mulan_laion/..."
                    # We need to convert to absolute path
                    filename = Path(file_path).name
                    full_path = root / filename
                    
                    if not full_path.exists():
                        # Try alternative: file_path might be relative from different base
                        full_path = Path(file_path)
                        if not full_path.exists():
                            continue
                    
                    # Select caption
                    if self.caption_type == "llava" and llava_caption and llava_caption != "N/A":
                        caption = llava_caption
                    else:
                        caption = blip2_caption if blip2_caption else ""
                    
                    # Group by image_id (include root name to avoid conflicts)
                    group_id = f"{root.name}_{image_id_num}"
                    image_groups[group_id]["layers"].append((layer_idx, str(full_path), caption))
        
        # Sort layers and filter by layer count
        filtered_groups = {}
        for image_id, data in image_groups.items():
            layers = data["layers"]
            layers.sort(key=lambda x: x[0])  # Sort by layer index
            
            if self.min_layers <= len(layers) <= self.max_layers:
                filtered_groups[image_id] = {
                    "paths": [item[1] for item in layers],
                    "captions": [item[2] for item in layers],
                }
        
        return filtered_groups
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[str], int, str]:
        image_id = self.image_ids[idx]
        group = self.image_groups[image_id]
        layer_paths = group["paths"]
        captions = group["captions"]
        
        # Load all layers
        layers = []
        for path in layer_paths:
            img = Image.open(path).convert('RGBA')
            img_tensor = self.transform(img)  # (4, H, W)
            layers.append(img_tensor)
        
        # Stack layers: (N, 4, H, W)
        layers_tensor = torch.stack(layers, dim=0)
        num_layers = len(layers)
        
        # Pad to max_layers if needed
        if num_layers < self.max_layers:
            padding = torch.zeros(
                self.max_layers - num_layers, 4, 
                self.resolution, self.resolution
            )
            layers_tensor = torch.cat([layers_tensor, padding], dim=0)
            # Pad captions with empty strings
            captions = captions + [""] * (self.max_layers - num_layers)
        
        return layers_tensor, captions, num_layers, image_id


def mulan_collate_fn(batch):
    """
    Custom collate function for MuLan dataset
    
    Returns:
        layers: (B, max_layers, 4, H, W)
        captions: List[List[str]] - (B, max_layers)
        num_layers: (B,) - actual layer counts
        image_ids: List[str]
    """
    layers = torch.stack([item[0] for item in batch], dim=0)
    captions = [item[1] for item in batch]  # List of Lists
    num_layers = torch.tensor([item[2] for item in batch], dtype=torch.long)
    image_ids = [item[3] for item in batch]
    
    return layers, captions, num_layers, image_ids


def build_mulan_dataloader(
    data_roots: List[str],
    batch_size: int = 4,
    resolution: int = 512,
    max_layers: int = 8,
    min_layers: int = 2,
    num_workers: int = 4,
    shuffle: bool = True,
    caption_type: str = "blip2",
) -> DataLoader:
    """
    Build MuLan DataLoader
    
    Args:
        data_roots: List of data directories (each should have meta_data.csv)
        batch_size: Batch size
        resolution: Image resolution
        max_layers: Maximum layers per image
        min_layers: Minimum layers per image
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        caption_type: "blip2" or "llava"
    
    Returns:
        DataLoader yielding (B, N, 4, H, W) tensors with captions
    """
    dataset = MuLanDataset(
        data_roots=data_roots,
        resolution=resolution,
        max_layers=max_layers,
        min_layers=min_layers,
        caption_type=caption_type,
    )    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mulan_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    return dataloader


# ============================================
# Reference-based Training 용 Helper
# ============================================
class MuLanReferenceBatch:
    """
    Reference-based architecture를 위한 batch 처리 helper
    
    각 sample에서 랜덤하게 target layer를 선택하고,
    나머지를 reference로 사용
    """
    
    @staticmethod
    def split_target_reference(
        layers: torch.Tensor,
        captions: List[List[str]],
        num_layers: torch.Tensor,
        target_layer_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Split layers into target and reference
        
        Args:
            layers: (B, N, 4, H, W)
            captions: List[List[str]] - (B, N)
            num_layers: (B,)
            target_layer_idx: If None, randomly select for each sample
        
        Returns:
            dict with:
                target_layer: (B, 4, H, W)
                target_caption: List[str] - (B,)
                reference_layers: (B, N-1, 4, H, W)
                reference_captions: List[List[str]] - (B, N-1)
                reference_mask: (B, N-1) - valid reference mask
        """
        B, N, C, H, W = layers.shape
        device = layers.device
        
        target_layers = []
        target_captions = []
        reference_layers = []
        reference_captions = []
        reference_masks = []
        
        for i in range(B):
            n = num_layers[i].item()
            
            # Select target layer index
            if target_layer_idx is not None:
                t_idx = min(target_layer_idx, n - 1)
            else:
                t_idx = torch.randint(0, n, (1,)).item()
            
            # Target
            target_layers.append(layers[i, t_idx])
            target_captions.append(captions[i][t_idx])
            
            # Reference (all except target)
            ref_indices = [j for j in range(N) if j != t_idx]
            ref_layer = layers[i, ref_indices]  # (N-1, 4, H, W)
            ref_caption = [captions[i][j] for j in ref_indices]
            
            # Reference mask (valid layers only)
            ref_mask = torch.zeros(N - 1, device=device)
            valid_refs = sum(1 for j in range(n) if j != t_idx)
            ref_mask[:valid_refs] = 1.0
            
            reference_layers.append(ref_layer)
            reference_captions.append(ref_caption)
            reference_masks.append(ref_mask)
        
        return {
            "target_layer": torch.stack(target_layers, dim=0),      # (B, 4, H, W)
            "target_caption": target_captions,                       # List[str]
            "reference_layers": torch.stack(reference_layers, dim=0), # (B, N-1, 4, H, W)
            "reference_captions": reference_captions,                # List[List[str]]
            "reference_mask": torch.stack(reference_masks, dim=0),   # (B, N-1)
        }


# ============================================
# Test code
# ============================================
if __name__ == "__main__":
    # Test with sample data
    data_roots = [
        "/data/mulan_coco",
        "/data/mulan_laion",
    ]
    
    dataloader = build_mulan_dataloader(
        data_roots=data_roots,
        batch_size=4,
        resolution=512,
        max_layers=8,
        num_workers=0,  # For debugging
        caption_type="blip2",
    )
    
    print(f"\nDataLoader created: {len(dataloader)} batches")
    
    # Test one batch
    for batch in dataloader:
        layers, captions, num_layers, image_ids = batch
        
        print(f"\n=== Raw Batch ===")
        print(f"Layers shape: {layers.shape}")  # (B, N, 4, H, W)
        print(f"Num layers: {num_layers}")
        print(f"Image IDs: {image_ids}")
        print(f"Sample captions[0]: {captions[0][:3]}...")  # First 3 captions of first sample
        
        # Test reference split
        print(f"\n=== Reference Split ===")
        split = MuLanReferenceBatch.split_target_reference(
            layers, captions, num_layers
        )
        print(f"Target layer shape: {split['target_layer'].shape}")
        print(f"Target captions: {split['target_caption']}")
        print(f"Reference layers shape: {split['reference_layers'].shape}")
        print(f"Reference mask: {split['reference_mask']}")
        
        break