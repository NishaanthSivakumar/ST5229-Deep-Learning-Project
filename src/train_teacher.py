"""Train a tiny supervised ViT classifier on CIFAR-10.

Its last-block CLS->patch attention is used as a semantic-importance signal
to drive semantic masking during MAE pretraining.

Design choice: we intentionally train this teacher only briefly (few epochs).
We do not need a state-of-the-art classifier, just attention that highlights
object regions more than background. This keeps the pipeline self-contained
(no external DINO download).
"""
import os, time, argparse, json
import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from data import get_cifar10_loaders
from model import ViTClassifier


def train_teacher(cfg, epochs=None, subset=None):
    epochs = epochs or cfg.teacher_epochs
    device = cfg.device
    train_loader, test_loader = get_cifar10_loaders(cfg, subset_train=subset, subset_test=subset)

    model = ViTClassifier(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.teacher_lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        running_loss, n, correct = 0., 0, 0
        pbar = tqdm(train_loader, desc=f"teacher ep {ep+1}/{epochs}", leave=False)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs)
            loss = crit(logits, lbls)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            correct += (logits.argmax(1) == lbls).sum().item()
            pbar.set_postfix(loss=running_loss/n, acc=correct/n)
        sched.step()

        test_acc = evaluate(model, test_loader, device)
        history.append({"epoch": ep+1, "train_loss": running_loss/n, "train_acc": correct/n,
                        "test_acc": test_acc, "time": time.time()-t0})
        print(f"[teacher] ep {ep+1}: train_loss={running_loss/n:.4f} train_acc={correct/n:.4f} test_acc={test_acc:.4f}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, "teacher.pt")
    torch.save({"model": model.state_dict(), "history": history}, ckpt_path)
    with open(os.path.join(cfg.results_dir, "teacher_history.json"), "w") as f:
        os.makedirs(cfg.results_dir, exist_ok=True)
        json.dump(history, f, indent=2)
    print(f"[teacher] saved -> {ckpt_path}")
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        correct += (logits.argmax(1) == lbls).sum().item()
        n += imgs.size(0)
    return correct / max(n, 1)


def load_teacher(cfg):
    model = ViTClassifier(cfg).to(cfg.device)
    ckpt = torch.load(os.path.join(cfg.ckpt_dir, "teacher.pt"), map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    cfg = get_config()
    train_teacher(cfg, epochs=args.epochs, subset=args.subset)
