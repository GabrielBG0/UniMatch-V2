# Reproducing UniMatch V2 Results

## Hardware

4 GPUs recommended to match paper numbers (standard PyTorch DDP). Fewer GPUs work but results may differ slightly.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Download Pre-trained DINOv2 Weights

Download from [Meta's DINOv2 repo](https://github.com/facebookresearch/dinov2) and place at:

```
./pretrained/dinov2_small.pth
./pretrained/dinov2_base.pth
./pretrained/dinov2_large.pth
```

Most configs default to `dinov2_small`.

## 3. Download Datasets

Update `data_root` in `configs/<dataset>.yaml` to point to your local dataset path.

| Dataset    | Size   | Notes                                           |
| ---------- | ------ | ----------------------------------------------- |
| PASCAL VOC | ~2 GB  | Easiest to start with                           |
| ADE20K     | ~4 GB  |                                                 |
| Cityscapes | ~11 GB | Requires registration at cityscapes-dataset.com |
| COCO       | ~20 GB |                                                 |

Train/val split files are already included in `splits/` — no need to generate them.

## 4. Run Training

Edit the top variables in `scripts/train.sh` to select dataset, method, and split:

```bash
dataset='pascal'        # pascal | cityscapes | ade20k | coco
method='unimatch_v2'   # unimatch_v2 | fixmatch | supervised
exp='dinov2_small'     # arbitrary label used in output path
split='366'            # see splits/<dataset>/ for available splits
```

Then launch:

```bash
sh scripts/train.sh 4 12345   # 4 GPUs, port 12345
```

Checkpoints are saved to `exp/<dataset>/<method>/<exp>/<split>/`. Training auto-resumes from `latest.pth` if interrupted.

## 5. SLURM

```bash
sh scripts/slurm_train.sh <num_gpu> <port> <partition>
```

## Remote Sensing (Change Detection)

Use the separate scripts in `remote-sensing/scripts/`.
