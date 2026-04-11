"""Central config for all experiments.

Small-scale setup so everything can run on CPU or a single small GPU.
CIFAR-10: 32x32 images, patch=4 -> 8x8 = 64 patches per image.
"""
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    # data
    data_root: str = "./data"
    img_size: int = 32
    patch_size: int = 4
    num_classes: int = 10
    batch_size: int = 256
    num_workers: int = 2

    # ViT backbone (ViT-Tiny-ish for CIFAR)
    embed_dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0

    # Decoder (lightweight)
    decoder_embed_dim: int = 96
    decoder_depth: int = 2
    decoder_num_heads: int = 3

    # MAE
    mask_ratio: float = 0.75
    masking_strategy: str = "random"   # "random" | "semantic_high" | "semantic_low"
    norm_pix_loss: bool = True

    # Training (pretraining)
    pretrain_epochs: int = 50
    pretrain_lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5

    # Teacher classifier (for semantic masking)
    teacher_epochs: int = 10
    teacher_lr: float = 5e-4

    # Linear probe
    probe_epochs: int = 20
    probe_lr: float = 1e-3

    seed: int = 42
    device: str = "cpu"  # auto-overridden at runtime if cuda available

    # Paths
    ckpt_dir: str = "./checkpoints"
    results_dir: str = "./results"

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

def get_config() -> Config:
    import torch
    cfg = Config()
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg
