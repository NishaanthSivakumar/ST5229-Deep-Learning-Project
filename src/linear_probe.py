"""Linear probe evaluation: freeze the MAE encoder, train a linear head on CLS token."""
import os, json, argparse, time
import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from data import get_cifar10_loaders
from model import MAE


def linear_probe(cfg, tag: str, epochs=None, subset=None):
    epochs = epochs or cfg.probe_epochs
    device = cfg.device
    train_loader, test_loader = get_cifar10_loaders(cfg, subset_train=subset, subset_test=subset)

    mae = MAE(cfg).to(device)
    ckpt = torch.load(os.path.join(cfg.ckpt_dir, f"mae_{tag}.pt"), map_location=device)
    mae.load_state_dict(ckpt["model"])
    mae.eval()
    for p in mae.parameters():
        p.requires_grad_(False)

    head = nn.Linear(cfg.embed_dim, cfg.num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.probe_lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    history = []
    for ep in range(epochs):
        head.train()
        t0, running, n, correct = time.time(), 0., 0, 0
        for imgs, lbls in tqdm(train_loader, desc=f"[probe {tag}] ep {ep+1}/{epochs}", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            with torch.no_grad():
                feats = mae.extract_features(imgs)
            logits = head(feats)
            loss = crit(logits, lbls)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * imgs.size(0)
            n += imgs.size(0)
            correct += (logits.argmax(1) == lbls).sum().item()
        sched.step()
        test_acc = _eval(head, mae, test_loader, device)
        history.append({"epoch": ep+1, "train_loss": running/n, "train_acc": correct/n,
                        "test_acc": test_acc, "time": time.time()-t0})
        print(f"[probe {tag}] ep {ep+1}: train_loss={running/n:.4f} train_acc={correct/n:.4f} test_acc={test_acc:.4f}")

    best = max(h["test_acc"] for h in history)
    final = history[-1]["test_acc"]
    os.makedirs(cfg.results_dir, exist_ok=True)
    out = {"tag": tag, "best_test_acc": best, "final_test_acc": final, "history": history}
    with open(os.path.join(cfg.results_dir, f"probe_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe {tag}] DONE best={best:.4f} final={final:.4f}")
    return out


@torch.no_grad()
def _eval(head, mae, loader, device):
    head.eval()
    correct, n = 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        feats = mae.extract_features(imgs)
        correct += (head(feats).argmax(1) == lbls).sum().item()
        n += imgs.size(0)
    return correct / max(n, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config()
    linear_probe(cfg, args.tag, epochs=args.epochs, subset=args.subset)
