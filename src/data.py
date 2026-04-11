"""CIFAR-10 data loaders."""
import os, torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def _train_tf():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

def _eval_tf():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

class _FakeCIFAR(Dataset):
    """Synthetic CIFAR-like dataset used only when the real dataset can't be
    downloaded (e.g., sandboxed environments). Each 'class' gets a distinct
    random colour prior so attention/classification still has signal.
    """
    def __init__(self, n=2048, num_classes=10, img_size=32, transform=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, num_classes, (n,), generator=g)
        priors = torch.rand(num_classes, 3, generator=g)  # class colour prior
        imgs = torch.rand(n, 3, img_size, img_size, generator=g) * 0.4
        for i in range(n):
            imgs[i] += priors[self.labels[i]].view(3, 1, 1) * 0.6
            # paint a bright square in a class-dependent location for "object"
            cx = 8 + (int(self.labels[i]) % 4) * 4
            cy = 8 + (int(self.labels[i]) // 4) * 4
            imgs[i, :, cy:cy+8, cx:cx+8] = 1.0
        self.imgs = imgs.clamp(0, 1)
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transform is not None:
            img = T.ToPILImage()(img)
            img = self.transform(img)
        return img, int(self.labels[i])


def _try_real_cifar(cfg):
    try:
        train_ds = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=_train_tf())
        test_ds  = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=_eval_tf())
        return train_ds, test_ds
    except Exception as e:
        print(f"[data] CIFAR-10 download unavailable ({type(e).__name__}); using synthetic fallback.")
        return None, None


def get_cifar10_loaders(cfg, subset_train: int = None, subset_test: int = None):
    use_fake = os.environ.get("SEMMAE_FAKE_DATA") == "1"
    train_ds = test_ds = None
    if not use_fake:
        train_ds, test_ds = _try_real_cifar(cfg)
    if train_ds is None:
        train_ds = _FakeCIFAR(n=2048, num_classes=cfg.num_classes, img_size=cfg.img_size, transform=_train_tf(), seed=0)
        test_ds  = _FakeCIFAR(n=512,  num_classes=cfg.num_classes, img_size=cfg.img_size, transform=_eval_tf(),  seed=1)

    if subset_train is not None:
        train_ds = Subset(train_ds, list(range(subset_train)))
    if subset_test is not None:
        test_ds = Subset(test_ds, list(range(subset_test)))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=False, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=False)
    return train_loader, test_loader

def denormalize(x):
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1).to(x.device)
    std  = torch.tensor(CIFAR_STD).view(1, 3, 1, 1).to(x.device)
    return (x * std + mean).clamp(0, 1)
