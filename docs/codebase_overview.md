# Codebase Overview

This document walks through how the UniMatch V2 codebase is structured and why things are laid out the way they are. Written for someone who already understands semi-supervised learning and wants to get productive quickly.

---

## The core idea in one paragraph

We have a small set of labeled images and a much larger set of unlabeled images. The model is trained simultaneously on both. For labeled images we use standard cross-entropy. For unlabeled images we use a teacher-student loop: a slow-moving EMA copy of the model (the teacher) generates pseudo-labels on a weakly-augmented view of each unlabeled image, and the student is trained to match those pseudo-labels on two *strongly*-augmented views of the same images. That is the FixMatch recipe. What we add on top is:

1. **Two strong views instead of one.** The student processes both simultaneously, which means we get two independent gradient signals per unlabeled image.
2. **CutMix on both strong views.** We paste a rectangular patch from a different image in the batch, and mirror the pseudo-label/confidence/ignore masks accordingly.
3. **Complementary Dropout.** When the student processes the two strong views as a single concatenated batch, we apply channel-level binary dropout masks to the intermediate DINOv2 features such that the two halves of the batch receive *complementary* masks — what one view drops, the other keeps. This forces the two views to learn from each other's blind spots.

---

## Directory layout

```
UniMatch-V2/
├── unimatch_v2.py      # main method (entry point)
├── fixmatch.py         # FixMatch baseline (entry point)
├── supervised.py       # supervised baseline + shared evaluate()
│
├── model/
│   ├── backbone/
│   │   ├── dinov2.py               # DINOv2 ViT implementation
│   │   └── dinov2_layers/          # attention, MLP, patch embed, etc.
│   ├── semseg/
│   │   └── dpt.py                  # DPT decoder head + full DPT model
│   └── util/
│       └── blocks.py               # FeatureFusionBlock, _make_scratch
│
├── dataset/
│   ├── semi.py                     # SemiDataset (labeled, unlabeled, val)
│   └── transform.py                # augmentation primitives
│
├── configs/
│   ├── pascal.yaml
│   ├── cityscapes.yaml
│   ├── ade20k.yaml
│   └── coco.yaml
│
├── splits/
│   ├── pascal/   {92, 183, 366, 732, 1464, all}/
│   ├── cityscapes/ {1_16, 1_8, 1_4, 1_2, all}/
│   ├── ade20k/   {1_64, 1_32, 1_16, 1_8, 1_4, 1_2, all}/
│   └── coco/     {1_512, 1_256, 1_128, 1_64, 1_32, all}/
│
├── util/
│   ├── dist_helper.py  # DDP setup
│   ├── utils.py        # AverageMeter, intersectionAndUnion, init_log
│   ├── ohem.py         # OHEM cross-entropy loss
│   └── classes.py      # per-dataset class name lists
│
├── scripts/
│   ├── train.sh        # launch with torch.distributed.launch
│   └── slurm_train.sh  # launch with srun
│
├── pretrained/         # DINOv2 weights go here (not committed)
├── training-logs/      # reference logs for all reported numbers
└── remote-sensing/     # self-contained sub-project for change detection
```

---

## The model: DINOv2 + DPT head

### Backbone (`model/backbone/dinov2.py`)

This is the original DINOv2 ViT code from Meta, mostly unmodified. The key method we use is `get_intermediate_layers(x, indices)`, which returns the token features (excluding the CLS token) from a chosen set of transformer blocks. We always pick 4 layers spread across the depth of the network:

| Size  | Depth | Layers tapped | Embed dim |
| ----- | ----- | ------------- | --------- |
| small | 12    | 2, 5, 8, 11   | 384       |
| base  | 12    | 2, 5, 8, 11   | 768       |
| large | 24    | 4, 11, 17, 23 | 1024      |
| giant | 40    | 9, 19, 29, 39 | 1536      |

Each returned tensor has shape `(B, N, C)` where `N = (H/14) * (W/14)` — DINOv2 uses a 14×14 patch size.

### DPT head (`model/semseg/dpt.py`)

`DPTHead` takes those 4 feature tensors and produces a segmentation map. The steps are:

1. **Reshape**: flatten spatial tokens back to 2D feature maps — `(B, N, C) → (B, C, H/14, W/14)`.
2. **Project**: 1×1 conv to bring each of the 4 feature maps to its target channel count.
3. **Rescale**: apply strided convolutions or transposed convolutions to align the 4 maps to different resolutions (×4, ×2, ×1, ×0.5 of the patch grid).
4. **Fuse bottom-up**: `scratch.layerX_rn` projects each map to a common `features` channel count; `FeatureFusionBlock` modules then fuse from deep to shallow — `refinenet4 → refinenet3 → refinenet2 → refinenet1`.
5. **Output**: a small 3×3 conv + 1×1 conv produces the per-class logits, which are then bilinearly upsampled back to the original image resolution.

The `DPT` model wraps backbone + head and handles Complementary Dropout in its `forward(x, comp_drop=True)` path. When `comp_drop=True`, the input is expected to be a batch of `2N` images (two strong views concatenated). Before feeding features into the head, we generate a random binary channel mask of shape `(N, C)` and assign the complement `2 - mask` to the second half. Any sample randomly selected as "kept" gets a mask of all-ones on both halves, so dropout isn't applied uniformly — roughly half the batch gets the complementary treatment and half passes through unchanged.

### Optimizer split

Backbone and head are given separate learning rates because DINOv2 is already a strong pretrained encoder — we fine-tune it slowly. The config key `lr_multi` (typically 40) scales the head LR relative to the backbone LR. AdamW with `weight_decay=0.01` is used for both groups.

---

## Data: splits, dataset, and augmentations

### Splits (`splits/`)

Each split directory has exactly two files: `labeled.txt` and `unlabeled.txt`. Each line is a space-separated pair: `relative/path/to/image.jpg relative/path/to/mask.png`. The paths are relative to `data_root` from the config.

`splits/<dataset>/val.txt` is always the full validation set, used regardless of which split is being trained.

When labeled data is scarce, `SemiDataset` repeats the labeled list until it matches the length of the unlabeled list. This keeps the two DataLoaders synchronized so we always consume both in lockstep.

### SemiDataset (`dataset/semi.py`)

One class handles all three modes:

- **`train_l`**: loads image + mask, applies random resize (0.5–2×), random crop to `crop_size`, horizontal flip, normalizes. Returns `(img, mask)`.
- **`train_u`**: loads image only (mask is a dummy zero tensor). Applies the same geometric augmentations, then produces three copies of the image: a weak view (`img_w`, no color distortion) and two strong views (`img_s1`, `img_s2`) with independent color jitter, grayscale, and Gaussian blur. Also returns two independently-sampled CutMix boxes and an `ignore_mask`. Returns `(img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2)`.
- **`val`**: normalizes only, returns `(img, mask, id)`.

One subtlety: the ignore mask for unlabeled images starts as all-zeros, but pixels that were in the padded border (value 254) get set to 255 before the tensor is returned. This propagates the "ignore this pixel" signal through the CutMix masking in the training loop.

### Augmentation primitives (`dataset/transform.py`)

All functions operate on PIL Images to keep things simple:

- `resize`: long-side resize with random scale from `ratio_range`.
- `crop`: random crop with padding if the image is smaller than `size`.
- `hflip`: horizontal flip with probability `p`.
- `blur`: Gaussian blur with random sigma in [0.1, 2.0].
- `normalize`: converts to tensor and applies ImageNet mean/std.
- `obtain_cutmix_box`: returns a binary `(H, W)` tensor with a random rectangle filled with 1. With probability `1-p` it returns an all-zero mask (no CutMix applied).

---

## The training loop

### Three entry points, same skeleton

All three scripts (`unimatch_v2.py`, `fixmatch.py`, `supervised.py`) follow the same structure:

1. Parse args, load YAML config.
2. Call `setup_distributed()` to initialize DDP via `torchrun`/`torch.distributed.launch`.
3. Build model, load DINOv2 pretrained weights into `model.backbone`, wrap in DDP.
4. Build `model_ema` as a `deepcopy` of the model with all gradients disabled.
5. Build dataloaders. The training loop zips `trainloader_l` and `trainloader_u` so each iteration sees one labeled batch and one unlabeled batch.
6. Check for `latest.pth` and resume automatically if it exists.
7. Train for `epochs` epochs, evaluate at the end of each, save `latest.pth` every epoch and `best.pth` when mIoU improves.

`supervised.py` also exports `evaluate()`, which both semi-supervised scripts import. This avoids duplicating the evaluation code.

### What unimatch_v2 does each iteration

```
# Teacher generates pseudo-labels (no grad)
pred_u_w = model_ema(img_u_w)
conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]   # confidence
mask_u_w = pred_u_w.argmax(dim=1)                   # class prediction

# Apply CutMix to both strong views (swap patches between images in the batch)
img_u_s1 = apply_cutmix(img_u_s1, cutmix_box1)
img_u_s2 = apply_cutmix(img_u_s2, cutmix_box2)

# Student forward (labeled + both strong views)
pred_x = model(img_x)
pred_u_s1, pred_u_s2 = model(cat(img_u_s1, img_u_s2), comp_drop=True).chunk(2)

# Labeled loss
loss_x = criterion_l(pred_x, mask_x)

# Unlabeled loss (only on high-confidence pixels, ignore border and cutmix-adjusted masks)
loss_u_s1 = masked_ce(pred_u_s1, mask_u_w_cutmixed1, conf >= conf_thresh)
loss_u_s2 = masked_ce(pred_u_s2, mask_u_w_cutmixed2, conf >= conf_thresh)

loss = (loss_x + (loss_u_s1 + loss_u_s2) / 2) / 2
```

CutMix pseudo-label adjustment mirrors the image operation: wherever `cutmix_box == 1`, the pseudo-label/confidence/ignore mask is taken from the flipped batch (`mask_u_w.flip(0)`), i.e., the corresponding image in the other half of the batch.

### FixMatch vs UniMatch V2

`fixmatch.py` is structurally identical but uses only one strong view (`img_u_s1`) and does not pass `comp_drop=True`. It also calls `torch.distributed.barrier()` before the optimizer step (an extra sync point not present in UniMatch V2).

### EMA update

At every iteration:

```python
ema_ratio = min(1 - 1 / (iters + 1), 0.996)
param_ema = param_ema * ema_ratio + param * (1 - ema_ratio)
```

The ratio ramps from 0 toward 0.996. Early in training the teacher tracks the student closely; as training progresses it becomes a very slow-moving average. Both parameters and buffers (e.g., BatchNorm running stats) are updated this way.

### Learning rate schedule

Polynomial decay with power 0.9 computed per iteration:

```python
lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
```

The backbone and head groups are updated together but with their independent base LR (`lr` and `lr * lr_multi`).

### Evaluation

`evaluate()` in `supervised.py` supports two modes:

- **`original`**: feeds each validation image at its original resolution and accumulates intersection/union stats.
- **`sliding_window`**: used for Cityscapes, which has very large images. Tiles the image with a stride of `crop_size // 2` and averages logits over overlapping patches before computing mIoU.

mIoU is computed with `intersectionAndUnion` in `util/utils.py` and reduced across all DDP ranks using `all_reduce`.

---

## Configuration (`configs/`)

Each YAML has the same keys:

| Key             | What it controls                                                         |
| --------------- | ------------------------------------------------------------------------ |
| `dataset`       | Dataset name string used by SemiDataset                                  |
| `data_root`     | Absolute path to the dataset on disk — **must be modified**              |
| `nclass`        | Number of output classes                                                 |
| `crop_size`     | Square crop size for training (518 for Pascal, 560 for Cityscapes, etc.) |
| `epochs`        | Total training epochs                                                    |
| `batch_size`    | Per-GPU batch size                                                       |
| `lr`            | Backbone learning rate                                                   |
| `lr_multi`      | Head LR multiplier (backbone LR × lr_multi = head LR)                    |
| `criterion`     | Loss function for labeled data: `CELoss` or `OHEM`                       |
| `conf_thresh`   | Confidence threshold below which pseudo-labels are ignored               |
| `backbone`      | One of `dinov2_small`, `dinov2_base`, `dinov2_large`                     |
| `lock_backbone` | If true, freeze backbone weights entirely during training                |

---

## Outputs and checkpointing

Each run saves to `exp/<dataset>/<method>/<exp>/<split>/`:

- `latest.pth`: full checkpoint saved every epoch (model, model_ema, optimizer, epoch, best mIoU).
- `best.pth`: copy of the checkpoint at the epoch with the highest student mIoU.
- `out.log`: stdout/stderr piped from the training script.
- TensorBoard event files: `train/loss_all`, `train/loss_x`, `train/loss_s`, `train/mask_ratio`, `eval/mIoU`, `eval/mIoU_ema`, per-class IoU.

If `latest.pth` exists when you launch training, the run resumes automatically. This is purely path-based — no command-line flag needed.

---

## Remote sensing sub-project (`remote-sensing/`)

This is a near-identical copy of the main codebase adapted for bi-temporal change detection on LEVIR-CD and WHU-CD. The key differences:

- Input is a **pair of images** (pre-change and post-change), not a single image.
- The dataset class is `SemiCDDataset` in `remote-sensing/dataset/semicd.py`.
- The model still uses DPT but outputs 2 classes (changed / unchanged).
- Evaluation reports changed-class IoU and overall accuracy (not mIoU over many classes).
- Recommended to train with a single GPU instead of 4.

Everything else — EMA teacher, CutMix, Complementary Dropout, checkpoint format, YAML config structure — is the same as the main codebase.

---

## Where to look for what

| You want to...                   | Go to                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| Change augmentation strength     | `dataset/transform.py` → `obtain_cutmix_box`, `blur`, `ColorJitter` params in `semi.py`  |
| Understand Complementary Dropout | `model/semseg/dpt.py` → `DPT.forward` when `comp_drop=True`                              |
| Understand pseudo-label masking  | `unimatch_v2.py` lines ~194–200                                                          |
| Add a new dataset                | Add a YAML config, add class names to `util/classes.py`, add a `val.txt` under `splits/` |
| Check a reported number          | `training-logs/` — every table cell has a matching log file                              |
| Get pretrained checkpoints       | HuggingFace: `LiheYoung/UniMatch-V2`                                                     |
