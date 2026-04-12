"""Pretrain MAE on CIFAR-10 with a configurable masking strategy.

Strategies:
  - random          : standard MAE random 75% masking
  - semantic_high   : mask top-k highest teacher-attention patches (semantic-guided)
  - semantic_low    : inverse ablation (mask background instead)

The teacher (small supervised ViT) must already be trained via train_teacher.py
when using a semantic strategy.
"""
import os, time, json, argparse, math
import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from data import get_cifar10_loaders
from model import MAE
from train_teacher import load_teacher


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def pretrain(cfg, strategy: str, epochs=None, subset=None, tag: str = None):
    epochs = epochs or cfg.pretrain_epochs
    tag = tag or strategy
    device = cfg.device
    train_loader, _ = get_cifar10_loaders(cfg, subset_train=subset)

    mae = MAE(cfg).to(device)
    opt = torch.optim.AdamW(mae.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    teacher = None
    if strategy.startswith("semantic"):
        teacher = load_teacher(cfg)
        for p in teacher.parameters():
            p.requires_grad_(False)

    total_steps = epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)
    step = 0
    history = []
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    for ep in range(epochs):
        mae.train()
        t0 = time.time()
        running, n = 0., 0
        pbar = tqdm(train_loader, desc=f"[{tag}] ep {ep+1}/{epochs}", leave=False)
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            lr = cosine_lr(step, total_steps, cfg.pretrain_lr, warmup_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            attn_map = None
            if teacher is not None:
                attn_map = teacher.cls_attention_map(imgs)  # (B, N)

            loss, _, _ = mae(imgs, strategy=strategy, attn_map=attn_map)
            opt.zero_grad(); loss.backward(); opt.step()

            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
            step += 1
            pbar.set_postfix(loss=running/n, lr=lr)

        ep_loss = running / n
        history.append({"epoch": ep+1, "loss": ep_loss, "time": time.time()-t0})
        print(f"[{tag}] ep {ep+1}/{epochs}: loss={ep_loss:.4f} ({time.time()-t0:.1f}s)")

    ckpt_path = os.path.join(cfg.ckpt_dir, f"mae_{tag}.pt")
    torch.save({"model": mae.state_dict(), "cfg": cfg.__dict__, "history": history, "strategy": strategy},
               ckpt_path)
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, f"mae_{tag}_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{tag}] saved -> {ckpt_path}")
    return mae, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["random", "semantic_high", "semantic_low"], default="random")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    cfg = get_config()
    pretrain(cfg, args.strategy, epochs=args.epochs, subset=args.subset, tag=args.tag)
