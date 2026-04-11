"""Produce qualitative figures:
  - Attention map from teacher overlaid on sample images
  - Mask pattern comparison: random vs semantic_high vs semantic_low
  - Reconstructions from each trained MAE
"""
import os, argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import get_config
from data import get_cifar10_loaders, denormalize
from model import MAE
from train_teacher import load_teacher


def _to_img(x):
    return denormalize(x).cpu().permute(0, 2, 3, 1).numpy()


def make_mask_image(imgs, mask, patch):
    """Return masked image where masked patches are grey."""
    B, C, H, W = imgs.shape
    gh = H // patch
    mask_img = denormalize(imgs).clone()
    mask_grid = mask.reshape(B, gh, gh)
    for b in range(B):
        for i in range(gh):
            for j in range(gh):
                if mask_grid[b, i, j] > 0.5:
                    mask_img[b, :, i*patch:(i+1)*patch, j*patch:(j+1)*patch] = 0.5
    return mask_img.cpu().permute(0, 2, 3, 1).numpy()


@torch.no_grad()
def run(cfg, n=6):
    device = cfg.device
    _, test_loader = get_cifar10_loaders(cfg)
    imgs, _ = next(iter(test_loader))
    imgs = imgs[:n].to(device)

    # Load teacher for attention maps
    teacher = load_teacher(cfg)
    attn = teacher.cls_attention_map(imgs)  # (B, N)
    gh = int(np.sqrt(attn.size(1)))
    attn_img = attn.reshape(-1, gh, gh).cpu().numpy()

    # Load each MAE
    strategies = ["random", "semantic_high", "semantic_low"]
    recons = {}
    masks = {}
    for s in strategies:
        ckpt_path = os.path.join(cfg.ckpt_dir, f"mae_{s}.pt")
        if not os.path.exists(ckpt_path):
            print(f"skip {s}: no ckpt")
            continue
        mae = MAE(cfg).to(device)
        mae.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        mae.eval()
        am = attn if s.startswith("semantic") else None
        _, pred, mask = mae(imgs, strategy=s, attn_map=am)
        recon = mae.unpatchify(pred)
        # combine: visible from original, masked from recon (display)
        recons[s] = _to_img(recon)
        masks[s] = make_mask_image(imgs, mask, cfg.patch_size)

    # ---- Figure 1: attention overlay ----
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
    orig = _to_img(imgs)
    for i in range(n):
        axes[0, i].imshow(orig[i]); axes[0, i].axis("off")
        axes[1, i].imshow(orig[i]); axes[1, i].imshow(attn_img[i], cmap="jet", alpha=0.5,
                                                        extent=(0, 32, 32, 0))
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title("Teacher CLS-attn", loc="left")
    fig.tight_layout()
    out1 = os.path.join(cfg.results_dir, "fig_teacher_attention.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 2: mask pattern comparison ----
    strategies_present = [s for s in strategies if s in masks]
    rows = 1 + len(strategies_present)
    fig, axes = plt.subplots(rows, n, figsize=(2*n, 2*rows))
    for i in range(n):
        axes[0, i].imshow(orig[i]); axes[0, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    for r, s in enumerate(strategies_present, start=1):
        for i in range(n):
            axes[r, i].imshow(masks[s][i]); axes[r, i].axis("off")
        axes[r, 0].set_title(s, loc="left")
    fig.tight_layout()
    out2 = os.path.join(cfg.results_dir, "fig_mask_patterns.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)

    # ---- Figure 3: reconstructions ----
    rows = 1 + 2 * len(strategies_present)
    fig, axes = plt.subplots(rows, n, figsize=(2*n, 2*rows))
    for i in range(n):
        axes[0, i].imshow(orig[i]); axes[0, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    for k, s in enumerate(strategies_present):
        for i in range(n):
            axes[1 + 2*k, i].imshow(masks[s][i]); axes[1 + 2*k, i].axis("off")
            axes[2 + 2*k, i].imshow(np.clip(recons[s][i], 0, 1)); axes[2 + 2*k, i].axis("off")
        axes[1 + 2*k, 0].set_title(f"{s} (masked)", loc="left")
        axes[2 + 2*k, 0].set_title(f"{s} (recon)", loc="left")
    fig.tight_layout()
    out3 = os.path.join(cfg.results_dir, "fig_reconstructions.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight"); plt.close(fig)

    print("saved:", out1, out2, out3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    cfg = get_config()
    os.makedirs(cfg.results_dir, exist_ok=True)
    run(cfg, n=args.n)
