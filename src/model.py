"""Minimal ViT and MAE implementation for CIFAR-10.

We keep this self-contained (no timm dependency) so it runs anywhere.
Shapes are tracked carefully for 32x32 images with 4x4 patches -> 64 patches.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- Positional encodings ----------------------
def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True):
    """Return (1, N(+1), C) 2D sin-cos positional embedding."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_w, grid_h, indexing='xy'), dim=0)  # (2, gh, gw)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    pos_embed = torch.cat([emb_h, emb_w], dim=1)  # (N, C)
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    return pos_embed.unsqueeze(0)  # (1, N(+1), C)

def _get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32) / (embed_dim / 2.)
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


# ---------------------- Transformer blocks ----------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_attn = None  # (B, heads, N, N) for inspection

    def forward(self, x, return_attn: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if return_attn:
            self.last_attn = attn.detach()
        attn_d = self.attn_drop(attn)
        out = (attn_d @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, return_attn=False):
        x = x + self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------- Patch embedding ----------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                       # (B, C, gh, gw)
        x = x.flatten(2).transpose(1, 2)        # (B, N, C)
        return x


# ---------------------- Supervised ViT (teacher) ----------------------
class ViTClassifier(nn.Module):
    """Tiny supervised ViT used as an attention teacher for semantic masking."""
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim)
        N = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(get_2d_sincos_pos_embed(cfg.embed_dim, int(math.sqrt(N)), cls_token=True),
                                      requires_grad=False)
        self.blocks = nn.ModuleList([Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x, return_attn=False):
        x = self.patch_embed(x)                                      # (B, N, C)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for i, blk in enumerate(self.blocks):
            x = blk(x, return_attn=return_attn and (i == len(self.blocks) - 1))
        x = self.norm(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats[:, 0])

    @torch.no_grad()
    def cls_attention_map(self, x):
        """Return CLS->patch attention averaged over heads from last block.
        Shape: (B, N_patches)
        """
        _ = self.forward_features(x, return_attn=True)
        attn = self.blocks[-1].attn.last_attn   # (B, H, N+1, N+1)
        cls_attn = attn[:, :, 0, 1:]             # CLS to patches
        return cls_attn.mean(dim=1)              # (B, N)


# ---------------------- MAE ----------------------
class MAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, 3, cfg.embed_dim)
        N = self.patch_embed.num_patches
        self.num_patches = N
        self.grid_size = self.patch_embed.grid_size
        self.patch_size = cfg.patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(get_2d_sincos_pos_embed(cfg.embed_dim, self.grid_size, cls_token=True),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(get_2d_sincos_pos_embed(cfg.decoder_embed_dim, self.grid_size, cls_token=True),
                                              requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(cfg.decoder_embed_dim, cfg.decoder_num_heads, cfg.mlp_ratio)
                                             for _ in range(cfg.decoder_depth)])
        self.decoder_norm = nn.LayerNorm(cfg.decoder_embed_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_embed_dim, cfg.patch_size * cfg.patch_size * 3, bias=True)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    # ---------- patchify / unpatchify ----------
    def patchify(self, imgs):
        p = self.patch_size
        h = w = self.grid_size
        x = imgs.reshape(imgs.size(0), 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(imgs.size(0), h * w, p * p * 3)
        return x  # (B, N, p*p*3)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = self.grid_size
        x = x.reshape(x.size(0), h, w, p, p, 3).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(x.size(0), 3, h * p, w * p)

    # ---------- masking ----------
    def _mask_from_scores(self, scores, mask_ratio):
        """Given per-patch scores (B, N), mask the top-k highest-score patches.
        Higher score == more likely to be masked.
        Returns ids_keep, ids_restore, mask (1=masked, 0=visible) and len_keep.
        """
        B, N = scores.shape
        len_keep = int(N * (1 - mask_ratio))
        ids_shuffle = torch.argsort(scores, dim=1, descending=False)  # low scores first -> KEEP low
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones(B, N, device=scores.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return ids_keep, ids_restore, mask, len_keep

    def random_masking(self, x, mask_ratio):
        B, N, _ = x.shape
        scores = torch.rand(B, N, device=x.device)
        return self._mask_from_scores(scores, mask_ratio)

    def semantic_masking(self, x, mask_ratio, attn_map, mode: str = "high"):
        """attn_map: (B, N) from teacher. mode='high' masks high-attention (semantic)
        patches preferentially; mode='low' inverts."""
        # We KEEP the patches with smallest scores; largest scores are MASKED.
        if mode == "high":
            scores = attn_map            # mask high-attention (semantic) patches
        elif mode == "low":
            scores = -attn_map           # inverse ablation: mask low-attention (background)
        else:
            raise ValueError(mode)
        # Add tiny noise to break ties stochastically
        scores = scores + 1e-6 * torch.randn_like(scores)
        return self._mask_from_scores(scores, mask_ratio)

    # ---------- encoder / decoder forward ----------
    def forward_encoder(self, imgs, mask_ratio, strategy="random", attn_map=None):
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]
        if strategy == "random":
            ids_keep, ids_restore, mask, _ = self.random_masking(x, mask_ratio)
        elif strategy == "semantic_high":
            assert attn_map is not None
            ids_keep, ids_restore, mask, _ = self.semantic_masking(x, mask_ratio, attn_map, mode="high")
        elif strategy == "semantic_low":
            assert attn_map is not None
            ids_keep, ids_restore, mask, _ = self.semantic_masking(x, mask_ratio, attn_map, mode="low")
        else:
            raise ValueError(strategy)

        x_kept = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x_kept.size(0), -1, -1)
        x_kept = torch.cat([cls, x_kept], dim=1)
        for blk in self.blocks:
            x_kept = blk(x_kept)
        x_kept = self.norm(x_kept)
        return x_kept, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B = x.size(0)
        mask_tokens = self.mask_token.expand(B, ids_restore.size(1) + 1 - x.size(1), -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(-1)))
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]  # drop cls

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.cfg.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)         # per-patch MSE
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs, strategy="random", attn_map=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, self.cfg.mask_ratio, strategy, attn_map)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # ---------- features for linear probe ----------
    @torch.no_grad()
    def extract_features(self, imgs):
        """Full forward (no masking) -> CLS token."""
        x = self.patch_embed(imgs) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]
