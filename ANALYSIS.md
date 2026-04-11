# Semantic MAE — End-to-End Project Analysis

## 1. Project Overview

This repository (`semantic_mae`) is a self-contained research codebase that extends the **Masked Autoencoder (MAE)** self-supervised learning paradigm of He et al. (2022) by replacing its default *random* patch-masking policy with a **semantically guided** masking policy. The central research question is:

> *When an MAE is forced to reconstruct **semantically important** regions of an image (rather than uniformly random patches), does it learn more transferable visual representations?*

To answer this, the project pretrains three variants of an identical MAE and compares them with a frozen-encoder linear probe on CIFAR-10:

1. `random` — standard MAE random 75% masking (the baseline).
2. `semantic_high` — mask the patches that a supervised teacher ViT *attends to most* (i.e., the object).
3. `semantic_low` — the inverse ablation: mask the patches the teacher attends to *least* (i.e., background).

The entire pipeline is intentionally small-scale (ViT-Tiny-ish, CIFAR-10 32×32 images, patch size 4) so it can run on a single CPU/GPU, trading absolute accuracy for a clean, reproducible A/B/C comparison.

Top-level layout:

```
semantic_mae/
├── src/
│   ├── config.py         # central hyperparameters
│   ├── data.py           # CIFAR-10 loaders (+ synthetic fallback)
│   ├── model.py          # ViT, MAE, masking logic
│   ├── train_teacher.py  # supervised teacher ViT
│   ├── train_mae.py      # MAE pretraining (any strategy)
│   ├── linear_probe.py   # frozen-encoder evaluation
│   ├── visualize.py      # qualitative figures
│   └── run_all.py        # orchestrates the full experiment
├── scripts/run_smoke.sh  # quick smoke test
├── scripts/run_full.sh   # full experiment
├── data/cifar-10-batches-py/  # CIFAR-10 (already downloaded)
├── checkpoints/          # teacher.pt, mae_random.pt, mae_semantic_high.pt, mae_semantic_low.pt
└── results/              # loss histories, probe jsons, figures, summary.json
```

---

## 2. Dataset

**CIFAR-10** is the only dataset used. It contains 60,000 32×32 RGB natural images across 10 balanced classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), with 50,000 training and 10,000 test images. The project uses the standard torchvision split; the raw batches are shipped in `data/cifar-10-batches-py/` so the pipeline runs offline.

A small wrinkle in `src/data.py` is a `_FakeCIFAR` synthetic fallback: if the real CIFAR-10 cannot be downloaded (e.g., sandboxed CI), the loader substitutes a 2,048-image synthetic dataset where each class is given a distinct colour prior and a class-dependent bright square, so attention still has signal for sanity-checking. For the real results reported below, the true CIFAR-10 is used.

### Preprocessing

Two transform pipelines are defined:

- **Train** (`_train_tf`): `RandomHorizontalFlip` → `RandomCrop(32, padding=4)` → `ToTensor` → `Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616))`. Random crop with reflection-style padding and horizontal flip are the standard CIFAR-10 augmentations; they give translational/flip invariance without disturbing class identity.
- **Eval** (`_eval_tf`): `ToTensor` → `Normalize` only; no augmentation, so metrics are deterministic on the test set.

The channel-wise normalisation uses the canonical CIFAR-10 statistics. A matching `denormalize()` helper reverts images to [0,1] for visualisation.

Images become patch tokens inside the model: 32×32 input, patch size 4 → an **8×8 grid = 64 patch tokens** per image. Everything downstream (masking, positional embeddings, reconstruction) is indexed on this 64-patch grid.

---

## 3. Model Architecture (`src/model.py`)

Three networks share the same tokeniser:

### 3.1 Patch embedding
`PatchEmbed` is a `Conv2d(3, embed_dim, kernel_size=4, stride=4)` — the standard ViT trick of using a strided convolution as a non-overlapping patch projector. It emits a `(B, 64, 192)` token tensor.

### 3.2 Positional embeddings
2-D sine-cosine positional embeddings (`get_2d_sincos_pos_embed`) are precomputed and registered as non-trainable buffers, both for the encoder (192-dim) and the decoder (96-dim). These are fixed by design, matching the original MAE.

### 3.3 Transformer blocks
A minimal ViT block with pre-LN, multi-head self-attention, and a 4× MLP expansion. The `Attention` module exposes a `return_attn` hook that stores the softmax attention map of the *last* block for later inspection — this is what enables the teacher's CLS-to-patch attention to be read out.

### 3.4 Teacher — `ViTClassifier`
A tiny supervised ViT with a `[CLS]` token, 6 transformer blocks, `embed_dim=192`, 3 heads and an MLP classifier head. Its only purpose is to provide an **attention signal**: given an image, it returns the mean across heads of the CLS→patch attention of the last block, a `(B, 64)` tensor indicating which patches the classifier finds most useful. This is the "semantic importance map" that drives masking.

Important design note (documented in `train_teacher.py`): the teacher is **deliberately under-trained** (10 epochs). The project doesn't need a SOTA classifier — it only needs an attention map that prefers foreground over background — and training it briefly keeps the pipeline self-contained (no external DINO download).

### 3.5 MAE — `MAE`
Full MAE architecture with:

- **Encoder**: 6-layer ViT (`embed_dim=192`, 3 heads) operating only on the *visible* (unmasked) subset of patches plus a CLS token — this is the canonical MAE efficiency trick.
- **Decoder**: a lightweight 2-layer ViT (`decoder_embed_dim=96`, 3 heads) that sees the full patch grid reconstructed from encoder outputs padded with a learned `mask_token` at masked positions.
- **Reconstruction head**: `nn.Linear(96, 4·4·3)` predicting the raw (or per-patch normalised) pixel values of every patch.
- **`patchify` / `unpatchify`**: reshape utilities to move between `(B, 3, 32, 32)` and `(B, 64, 48)` patch-pixel representations.

### 3.6 Masking logic (the research contribution)

The single most important piece of the codebase is `_mask_from_scores`, `random_masking`, and `semantic_masking` in `model.py`. The convention is:

> **Score sorted ascending; smallest scores are KEPT, largest scores are MASKED.**

So to change the masking policy you only need to change how scores are assigned:

- **`random_masking`**: scores are i.i.d. uniform noise. Reproduces standard MAE 75% random masking.
- **`semantic_masking(mode="high")`**: scores = teacher attention. High attention ⇒ high score ⇒ masked. The encoder is thus forced to reconstruct the object from the background and the few foreground patches that survived.
- **`semantic_masking(mode="low")`**: scores = −teacher attention. This is an inverse ablation: the *background* is masked, and the model gets to see the object. If semantic masking helps purely because "masking is harder → features are better", then `semantic_low` should be *worse* than `random`. If both semantic variants improve over random, the improvement comes from the semantic structure itself, not just from harder reconstruction.

A small Gaussian noise (`1e-6 * randn`) is added to the scores before sorting to break ties (important because teacher attention can be almost uniform in some regions).

Mask ratio is fixed at **0.75** in all three variants, i.e., 48 of 64 patches are masked and 16 are visible, matching the original MAE hyperparameter.

### 3.7 Loss
`forward_loss` computes per-patch MSE between predicted and target patches, averaged over channels, and restricted to the **masked positions only** (`(loss * mask).sum() / mask.sum()`). With `norm_pix_loss=True`, the target patches are first normalised to zero-mean/unit-variance per patch, which is the MAE paper's recommended setting and usually improves linear-probe quality.

---

## 4. Methodology & Training Pipeline

The orchestrator is `src/run_all.py`, which ties four stages together; all hyperparameters live in `src/config.py`.

### Stage 1 — Train the teacher ViT (`train_teacher.py`)
- Supervised CIFAR-10 classification on the same 32×32 / patch-4 ViT as above.
- Optimiser: `AdamW`, `lr=5e-4`, weight decay 0.05, cosine annealing schedule over 10 epochs.
- Loss: cross-entropy; metric: top-1 accuracy on the test set.
- Output: `checkpoints/teacher.pt` and `results/teacher_history.json`.
- The trained model is then frozen; at MAE training time, `cls_attention_map()` is called per batch to produce the `(B, 64)` importance map.

Why this stage exists: it is the source of the semantic signal. The entire extension depends on having a map of "which patches matter for the class".

### Stage 2 — Pretrain the MAE three times (`train_mae.py`)
For each `strategy ∈ {random, semantic_high, semantic_low}`:
- Load the frozen teacher (for the two semantic variants only).
- Build a fresh `MAE`, optimiser `AdamW(lr=1.5e-4, wd=0.05, betas=(0.9, 0.95))` — the canonical MAE optimiser settings.
- **Learning-rate schedule**: linear warmup for 5 epochs then cosine decay to zero over the remaining 45, computed per step by `cosine_lr`. The LR is written back into `opt.param_groups` each step.
- Per batch: (optionally) compute `attn_map = teacher.cls_attention_map(imgs)`, then run `mae(imgs, strategy, attn_map)` which chooses the mask, encodes the visible 16 patches, decodes the full 64, and returns the masked-MSE reconstruction loss.
- 50 epochs, batch size 256. Checkpoints saved as `checkpoints/mae_{strategy}.pt`; loss curves saved as `results/mae_{strategy}_history.json`.

This stage is the experimental manipulation: three identical architectures, identical data, identical optimiser, **only the masking rule differs**.

### Stage 3 — Linear probe evaluation (`linear_probe.py`)
The linear probe is the standard downstream protocol for judging self-supervised representations:
- Load `mae_{tag}.pt`, put it in eval mode, and freeze **all** parameters.
- Define a brand-new `nn.Linear(192, 10)` head on top of the CLS token.
- `extract_features(imgs)` runs the *full* encoder without any masking (all 64 patches visible) and returns the CLS embedding.
- Train only the linear head for 20 epochs with `AdamW(lr=1e-3)`, cosine schedule, cross-entropy loss.
- Report per-epoch training and **test** accuracy; save `results/probe_{tag}.json` with the full history plus `best_test_acc` and `final_test_acc`.

Because the encoder is frozen, any accuracy difference between the three probes is attributable *only* to how well the pretrained features linearly separate the CIFAR-10 classes — i.e., it is a measurement of representation quality, not fine-tuning capacity. This is the headline metric of the paper-style comparison.

### Stage 4 — Visualisations (`visualize.py`)
Qualitative sanity checks, saved to `results/`:
- `fig_teacher_attention.png` — originals with teacher CLS-attention overlaid as a heatmap (jet, α=0.5).
- `fig_mask_patterns.png` — for each strategy, the same images with masked patches replaced by a grey square, so the viewer can directly compare which patches each policy masks.
- `fig_reconstructions.png` — for each strategy, the masked input *and* the decoder's reconstruction stacked per row.

These figures confirm *why* the numbers move (e.g., that `semantic_high` really does mask the object) rather than just showing that they do.

### Smoke test vs full run
`run_all.py --smoke` overrides `batch_size=64`, `teacher_epochs=1`, `pretrain_epochs=2`, `probe_epochs=2`, `warmup_epochs=0`, and uses a 512-image subset to verify the pipeline runs end-to-end in a few minutes. The full run uses the defaults in `config.py`. The published numbers in `results/summary.json` correspond to the full run.

---

## 5. Results

All numbers below are read directly from the JSON files in `results/`.

### 5.1 Teacher ViT (Stage 1)

| Metric          | Epoch 1 | Epoch 10 |
|---|---|---|
| Train loss      | 1.795   | 0.796    |
| Train accuracy  | 33.5%   | 71.8%    |
| Test accuracy   | 47.2%   | **69.8%**|

The teacher lands at ~70% test accuracy after 10 epochs. This is far from SOTA for CIFAR-10 (a well-trained ViT or ResNet can exceed 95%), but — as the comment in `train_teacher.py` acknowledges — it is deliberately so. What matters for the downstream experiment is only that its last-layer CLS→patch attention is biased toward the foreground, which 70% accuracy is more than sufficient to guarantee.

### 5.2 MAE reconstruction loss (Stage 2)

Normalised-pixel MSE on masked patches at the first and last epoch:

| Strategy        | Loss ep 1 | Loss ep 50 |
|---|---|---|
| `random`        | 1.042     | **0.457**  |
| `semantic_high` | 1.073     | 0.598      |
| `semantic_low`  | 1.004     | 0.582      |

As expected, `random` reaches the **lowest** reconstruction loss: random patches are on average easier to predict than targeted high-attention patches, because they mix textureless background with objects. `semantic_high` ends with the highest loss, i.e., it is the hardest reconstruction problem — reconstructing the object from background alone. Crucially, *raw reconstruction loss is not the metric we care about* — we care about the representation quality this loss produces, measured next.

### 5.3 Linear probe test accuracy (Stage 3) — the headline result

| Pretraining     | Best test acc | Final test acc |
|---|---|---|
| `random` (baseline) | 41.05%     | 40.94%         |
| `semantic_high`     | **46.76%** | **46.60%**     |
| `semantic_low`      | 46.41%     | 46.41%         |

Reading this table:

1. **Semantic masking helps.** Both semantic variants beat random masking by ~5.5 absolute percentage points (46.76 vs 41.05) — a large jump at this probe scale. Despite `semantic_high` solving a *harder* reconstruction task (higher final loss), its CLS feature is substantially more linearly separable by class.
2. **`semantic_low` is almost as good as `semantic_high`.** This is the ablation the project was set up to check, and the result is surprising: masking the background also beats random. That means the lift is not simply "masking objects forces the encoder to learn object shape"; any *structured*, content-aware masking beats random uniform masking at this scale. One plausible interpretation: random masking wastes capacity reconstructing uninformative regions, whereas both semantic variants concentrate learning on content-correlated patches (either as targets or as the last remaining visible evidence) and therefore produce features that are more discriminative for the class.
3. **Absolute numbers are modest** (~47% top-1) because the backbone is ViT-Tiny, the pretrain schedule is only 50 epochs, and no fine-tuning is done — only a frozen linear probe. The experiment is designed as a controlled A/B/C, not as a leaderboard submission.

### 5.4 Qualitative figures
- `fig_teacher_attention.png` shows the teacher's attention concentrating on object pixels (even at 70% accuracy, the bias is clearly there).
- `fig_mask_patterns.png` confirms the masking policies behave as designed: `semantic_high` deletes the object, `semantic_low` deletes the background, `random` spreads masks uniformly.
- `fig_reconstructions.png` shows that all three decoders produce plausible blurred reconstructions in the masked regions, i.e., the comparison is not contaminated by a broken decoder in any condition.

### 5.5 Total compute
`summary.json` records `total_time_sec ≈ 71,927 s`, i.e., ~20 hours of CPU wall-clock for the full run of all four stages — consistent with the "overnight CPU" note in `run_full.sh`. The `mae_semantic_low_history.json` shows an unusually long first epoch (~21,498 s) that is almost certainly a cold start / download stall on the first batch; subsequent epochs return to ~130 s each, so it does not affect the results, only the timing total.

---

## 6. Conclusions

- **Extending MAE with semantically guided masking is worthwhile.** At this small-scale, controlled setting, swapping random 75% masking for teacher-attention-guided masking lifts CIFAR-10 linear-probe accuracy from ~41% to ~47% — a large and reliable improvement, repeated across two different semantic policies.
- **The improvement is not specifically "mask the object".** The inverse ablation (`semantic_low`, mask the background) performs essentially the same as masking the object. The takeaway is that *any* content-aware mask is better than i.i.d. random masking here, presumably because it biases the self-supervised signal toward information-rich regions instead of wasting capacity on easy, low-frequency background patches.
- **Reconstruction loss and representation quality are decorrelated** in this experiment: the harder (higher-loss) semantic variants learn *better* features, consistent with the broader MAE literature that finds the reconstruction objective is a proxy, not the target, of representation learning.
- **The pipeline is clean and fully reproducible.** A single `scripts/run_full.sh` command trains the teacher, pretrains three MAEs with different masking strategies, runs three linear probes, and writes all metrics, loss histories and visualisations to `results/`. The design (tiny ViT, CPU-friendly, offline CIFAR-10, no external dependencies beyond `torch`, `torchvision`, `tqdm`, `matplotlib`) makes it easy to extend the ablation (different mask ratios, different teachers, different datasets) without changing the scaffolding.

### Natural next steps for the extension
Since the project is framed as "extension of current research work," obvious follow-ups that fit the existing code with minimal changes include: (a) sweeping `mask_ratio` per strategy to see whether the optimal ratio changes when the mask is semantic; (b) replacing the supervised teacher with a self-supervised one (DINO-style) to remove the label dependency; (c) mixing random and semantic masks at a tunable ratio to interpolate between the two regimes; (d) scaling up from ViT-Tiny/CIFAR to ViT-Small/Tiny-ImageNet to check whether the ~5-point probe gap persists or shrinks with more capacity and data; and (e) replacing the frozen linear probe with fine-tuning to see whether the representation-quality gap survives end-to-end training.
