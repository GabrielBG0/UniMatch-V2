# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

UniMatch V2 is a PyTorch implementation of semi-supervised semantic segmentation, published in TPAMI 2025. It uses DINOv2 ViT encoders with a DPT decoder head, and applies consistency regularization between two strongly-augmented views of unlabeled images (via CutMix + Complementary Dropout). There is also a `remote-sensing/` sub-project for semi-supervised change detection.

## Training

```bash
# Standard multi-GPU training (4 GPUs recommended to reproduce paper results)
sh scripts/train.sh <num_gpu> <port>

# SLURM
sh scripts/slurm_train.sh <num_gpu> <port> <partition>
```

To switch datasets, splits, or methods, edit the variables at the top of `scripts/train.sh`:
- `dataset`: `pascal` | `cityscapes` | `ade20k` | `coco`
- `method`: `unimatch_v2` | `fixmatch` | `supervised`
- `split`: see `splits/<dataset>/` for available directories (e.g., `92`, `366`, `1_16`)
- `exp`: arbitrary string used as part of the output path

Output checkpoints go to `exp/<dataset>/<method>/<exp>/<split>/`. Training auto-resumes from `latest.pth` if it exists there.

For remote sensing change detection, use the separate scripts in `remote-sensing/scripts/`.

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Place pre-trained DINOv2 weights at:
```
./pretrained/dinov2_small.pth
./pretrained/dinov2_base.pth
./pretrained/dinov2_large.pth
```

Update the `data_root` field in the appropriate `configs/<dataset>.yaml` to point to your dataset location.

## Architecture

**Entry points** — each is a self-contained training script:
- `unimatch_v2.py` — main semi-supervised method
- `fixmatch.py` — FixMatch baseline
- `supervised.py` — fully-supervised baseline; also exports the shared `evaluate()` function

**Model** (`model/`):
- `model/backbone/dinov2.py` — DINOv2 ViT backbone; `get_intermediate_layers()` returns features from 4 specified transformer layers
- `model/semseg/dpt.py` — `DPT` model: backbone + `DPTHead`. The head projects 4 intermediate feature maps to multi-scale outputs, then fuses them bottom-up via `FeatureFusionBlock`. `DPT.forward(x, comp_drop=True)` applies Complementary Dropout: channel-wise binary masks are generated in complementary pairs so the two augmented views receive opposite dropped channels.
- `model/util/blocks.py` — `FeatureFusionBlock`, `_make_scratch` used by DPTHead

**Data** (`dataset/`):
- `dataset/semi.py` — `SemiDataset`: unified dataset for labeled (`train_l`), unlabeled (`train_u`), and validation splits. Unlabeled mode returns `(img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2)` — weak view + two independently augmented strong views + CutMix boxes.
- `dataset/transform.py` — augmentation primitives used by SemiDataset

**Splits** (`splits/<dataset>/<split>/`): Each split directory contains `labeled.txt` and `unlabeled.txt` listing image/mask path pairs. `splits/<dataset>/val.txt` is always used for validation.

**Config** (`configs/<dataset>.yaml`): Controls `data_root`, `nclass`, `crop_size`, `epochs`, `batch_size`, `lr`, `lr_multi`, `conf_thresh`, `backbone` (`dinov2_small/base/large`), `lock_backbone`, and `criterion`.

**Key training detail**: The EMA model (`model_ema`) generates pseudo-labels on the weak view; the student model trains on two CutMix-augmented strong views with Complementary Dropout (`comp_drop=True`). Pseudo-labels are filtered by `conf_thresh`. LR follows a polynomial decay schedule. The backbone and head use separate LR groups scaled by `lr_multi` (typically 40×).

**Backbone sizes** and their DPT feature configs:
| Size   | features | out_channels            | Intermediate layers |
|--------|----------|-------------------------|---------------------|
| small  | 64       | [48, 96, 192, 384]      | [2, 5, 8, 11]       |
| base   | 128      | [96, 192, 384, 768]     | [2, 5, 8, 11]       |
| large  | 256      | [256, 512, 1024, 1024]  | [4, 11, 17, 23]     |
| giant  | 384      | [1536, 1536, 1536, 1536]| [9, 19, 29, 39]     |

## Utilities

- `util/dist_helper.py` — `setup_distributed()` initializes DDP; reads `LOCAL_RANK` from env
- `util/utils.py` — `AverageMeter`, `intersectionAndUnion`, `init_log`, `count_params`
- `util/ohem.py` — OHEM cross-entropy loss (used for some dataset configs)
- `util/classes.py` — per-dataset class name lists used in logging
