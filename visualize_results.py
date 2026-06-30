#!/usr/bin/env python3
"""
Visualize UniMatch V2 experiment results from TensorBoard event files.

Run from the project root:
    python visualize_results.py

Outputs (PNG + PDF) in figures/:
  1_miou_vs_labeled   -- mIoU vs. labeled samples, small vs. base backbone
  2_training_curves   -- mIoU over epochs for all splits (small backbone)
  3_per_class_iou     -- per-class IoU bar chart for best run per dataset
  4_loss_curves       -- smoothed training loss for all splits (small backbone)
"""

import os
import glob
import re
from collections import defaultdict

import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorboard.backend.event_processing import event_accumulator

# ── config ───────────────────────────────────────────────────────────────────
EXP_DIR    = 'exp'
SPLITS_DIR = 'splits'
FIG_DIR    = 'figures'
METHOD     = 'unimatch_v2'

DATASETS = ['cityscapes', 'ade20k', 'coco']

DATASET_LABEL = {
    'cityscapes': 'Cityscapes (19 cls)',
    'ade20k':     'ADE20K (150 cls)',
    'coco':       'COCO (81 cls)',
}

BACKBONE_STYLE = {
    'dinov2_small': dict(marker='o', linestyle='-',  color='steelblue',  label='DINOv2-S'),
    'dinov2_base':  dict(marker='s', linestyle='--', color='darkorange', label='DINOv2-B'),
}

PAPER_BACKBONE_STYLE = {
    'Small': dict(marker='o', linestyle=':',  color='steelblue'),
    'Base':  dict(marker='s', linestyle=':', color='darkorange'),
}

PAPER_RESULTS_CSV = 'paper_results.csv'

# dataset name as it appears in the CSV vs internal key
PAPER_DATASET_KEY = {
    'ADE20K':     'ade20k',
    'Cityscapes': 'cityscapes',
    'COCO':       'coco',
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_paper_results():
    """Return {dataset_key: {backbone_label: [(frac_val, miou), ...]}}."""
    results = defaultdict(lambda: defaultdict(list))
    if not os.path.exists(PAPER_RESULTS_CSV):
        return results
    with open(PAPER_RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            dataset_key = PAPER_DATASET_KEY.get(row['Dataset'])
            if dataset_key is None:
                continue
            num_s, den_s = row['Labeled Regime'].split('/')
            frac_val = int(num_s) / int(den_s)
            results[dataset_key][row['Backbone']].append((frac_val, float(row['mIoU'])))
    return results


def load_scalars(run_dir):
    """Return {tag: [(step, value), ...]} for a run directory."""
    tf_files = glob.glob(os.path.join(run_dir, '*.tfevents*'))
    if not tf_files:
        return {}
    # Use the most recent file if multiple exist (e.g. resumed run)
    tf_files.sort(key=os.path.getmtime)
    ea = event_accumulator.EventAccumulator(tf_files[-1], size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in tags}


def count_labeled(dataset, split):
    path = os.path.join(SPLITS_DIR, dataset, split, 'labeled.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return sum(1 for _ in f)


def parse_fraction(split):
    """Convert split name like '1_16' to (numerator, denominator, float)."""
    m = re.match(r'^(\d+)_(\d+)$', split)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return num, den, num / den
    return None, None, None


def fraction_label(num, den):
    if num == 1:
        return f'1/{den}'
    return f'{num}/{den}'


def discover_experiments():
    """Walk exp/ and return a list of experiment dicts."""
    exps = []
    for run_dir in sorted(glob.glob(os.path.join(EXP_DIR, '*', METHOD, '*', '*'))):
        parts = run_dir.replace('\\', '/').split('/')
        # parts: exp / dataset / method / backbone / split
        if len(parts) < 5:
            continue
        dataset, backbone, split = parts[-4], parts[-2], parts[-1]
        if dataset not in DATASETS:
            continue
        if not glob.glob(os.path.join(run_dir, '*.tfevents*')):
            continue
        frac_num, frac_den, frac_val = parse_fraction(split)
        exps.append(dict(
            dataset=dataset, backbone=backbone, split=split,
            run_dir=run_dir,
            n_labeled=count_labeled(dataset, split),
            frac_num=frac_num, frac_den=frac_den, frac_val=frac_val,
        ))
    return exps


def best_miou_ema(scalars):
    for key in ('eval/mIoU_ema', 'eval/mIoU'):
        if key in scalars and scalars[key]:
            return max(v for _, v in scalars[key])
    return None


def per_class_ious_at_best(scalars):
    """Return {class: iou} at the epoch that achieved the best mIoU_ema."""
    ema_key = 'eval/mIoU_ema' if 'eval/mIoU_ema' in scalars else 'eval/mIoU'
    if ema_key not in scalars:
        return {}
    best_step = max(scalars[ema_key], key=lambda sv: sv[1])[0]
    result = {}
    for tag, vals in scalars.items():
        m = re.match(r'eval/(.+)_IoU_ema$', tag)
        if not m:
            continue
        # Find value at or nearest to best_step
        nearest = min(vals, key=lambda sv: abs(sv[0] - best_step))
        result[m.group(1)] = nearest[1]
    return result


def smooth(values, window_frac=0.02):
    w = max(1, int(len(values) * window_frac))
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='same')


# ── figure 1: mIoU vs. labeled samples ───────────────────────────────────────

def fig_miou_vs_labeled(exps, paper_results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('UniMatch V2 Reproduction — Best mIoU vs. Labeled Fraction', fontsize=13, fontweight='bold')

    # backbone label used in paper CSV -> internal backbone key
    paper_to_internal = {'Small': 'dinov2_small', 'Base': 'dinov2_base'}

    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(DATASET_LABEL[dataset], fontsize=11)
        ax.set_xlabel('Fraction of labeled data')
        ax.set_ylabel('Best mIoU (EMA)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')

        all_fracs = {}  # frac_val -> label string for tick labeling
        plotted_any = False

        # ── our runs ──────────────────────────────────────────────────────────
        for backbone, style in BACKBONE_STYLE.items():
            pts = []
            for e in exps:
                if e['dataset'] != dataset or e['backbone'] != backbone:
                    continue
                if e['frac_val'] is None:
                    continue
                scalars = load_scalars(e['run_dir'])
                miou = best_miou_ema(scalars)
                if miou is not None:
                    pts.append((e['frac_val'], miou))
                    all_fracs[e['frac_val']] = fraction_label(e['frac_num'], e['frac_den'])
            if not pts:
                continue
            pts.sort()
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=style['marker'], linestyle=style['linestyle'],
                    color=style['color'], label=style['label'],
                    linewidth=1.8, markersize=6)
            for x, y in zip(xs, ys):
                ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                            xytext=(0, 7), ha='center', fontsize=7.5, color=style['color'])
            plotted_any = True

        # ── paper reference ───────────────────────────────────────────────────
        for paper_bb, pstyle in PAPER_BACKBONE_STYLE.items():
            pts = sorted(paper_results[dataset].get(paper_bb, []))
            if not pts:
                continue
            xs, ys = zip(*pts)
            for fv in xs:
                num_s, den_s = 1, round(1 / fv)
                all_fracs[fv] = fraction_label(num_s, den_s)
            internal_bb = paper_to_internal[paper_bb]
            base_style = BACKBONE_STYLE[internal_bb]
            ax.plot(xs, ys, marker=pstyle['marker'], linestyle=pstyle['linestyle'],
                    color=pstyle['color'], alpha=0.6,
                    label=f"{base_style['label']} (paper)",
                    linewidth=1.4, markersize=5)
            plotted_any = True

        if all_fracs:
            sorted_fracs = sorted(all_fracs.keys())
            ax.xaxis.set_major_locator(mticker.FixedLocator(sorted_fracs))
            ax.xaxis.set_major_formatter(mticker.FixedFormatter([all_fracs[f] for f in sorted_fracs]))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.tick_params(axis='x', which='major', labelsize=8)
        if plotted_any:
            ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ── figure 2: mIoU training curves ───────────────────────────────────────────

def fig_training_curves(exps):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('UniMatch V2 — mIoU Training Curves (DINOv2-S, EMA)', fontsize=13, fontweight='bold')

    cmap = plt.get_cmap('viridis')

    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(DATASET_LABEL[dataset], fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIoU EMA')
        ax.grid(True, alpha=0.3, linestyle='--')

        runs = sorted(
            [e for e in exps if e['dataset'] == dataset and e['backbone'] == 'dinov2_small'],
            key=lambda e: e['n_labeled'] or 0,
        )
        n = len(runs)
        for i, e in enumerate(runs):
            scalars = load_scalars(e['run_dir'])
            key = 'eval/mIoU_ema' if 'eval/mIoU_ema' in scalars else 'eval/mIoU'
            if key not in scalars:
                continue
            steps, vals = zip(*scalars[key])
            color = cmap(i / max(n - 1, 1))
            ax.plot(steps, vals, color=color, linewidth=1.6,
                    label=f"{e['split']}  ({e['n_labeled']} lbl)")

        ax.legend(fontsize=7.5, loc='lower right')

    fig.tight_layout()
    return fig


# ── figure 3: per-class IoU ───────────────────────────────────────────────────

def fig_per_class_iou(exps):
    # One panel per dataset; use the run with most labeled data (most stable)
    candidates = [
        e for e in exps
        if e['backbone'] == 'dinov2_small' and e['dataset'] in DATASETS
    ]
    best_runs = {}
    for e in candidates:
        d = e['dataset']
        if d not in best_runs or (e['n_labeled'] or 0) > (best_runs[d]['n_labeled'] or 0):
            best_runs[d] = e

    n_panels = sum(1 for d in DATASETS if d in best_runs)
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle('UniMatch V2 — Per-Class IoU at Best Epoch (DINOv2-S)', fontsize=13, fontweight='bold')

    ax_idx = 0
    for dataset in DATASETS:
        if dataset not in best_runs:
            continue
        ax = axes[ax_idx]; ax_idx += 1
        e = best_runs[dataset]
        scalars = load_scalars(e['run_dir'])
        class_ious = per_class_ious_at_best(scalars)
        if not class_ious:
            ax.set_visible(False)
            continue

        classes = sorted(class_ious, key=lambda c: class_ious[c])
        vals    = [class_ious[c] for c in classes]
        colors  = ['#d62728' if v < 40 else '#ff7f0e' if v < 60 else '#2ca02c' for v in vals]

        ax.barh(classes, vals, color=colors, edgecolor='white', height=0.75)
        mean_iou = np.mean(vals)
        ax.axvline(mean_iou, color='steelblue', linestyle='--', linewidth=1.5,
                   label=f'mean {mean_iou:.1f}%')
        ax.set_xlim(0, 100)
        ax.set_xlabel('IoU (%)')
        ax.set_title(f"{DATASET_LABEL[dataset]}  [{e['split']}]", fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(axis='y', labelsize=max(5, 9 - len(classes) // 15))

    fig.tight_layout()
    return fig


# ── figure 4: loss curves ─────────────────────────────────────────────────────

def fig_loss_curves(exps):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle('UniMatch V2 — Training Loss (DINOv2-S, smoothed)', fontsize=13, fontweight='bold')

    cmap = plt.get_cmap('plasma')

    for ax, dataset in zip(axes, DATASETS):
        ax.set_title(DATASET_LABEL[dataset], fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('Total loss')
        ax.grid(True, alpha=0.3, linestyle='--')

        runs = sorted(
            [e for e in exps if e['dataset'] == dataset and e['backbone'] == 'dinov2_small'],
            key=lambda e: e['n_labeled'] or 0,
        )
        n = len(runs)
        for i, e in enumerate(runs):
            scalars = load_scalars(e['run_dir'])
            if 'train/loss_all' not in scalars:
                continue
            steps, vals = zip(*scalars['train/loss_all'])
            smoothed = smooth(np.array(vals))
            color = cmap(i / max(n - 1, 1))
            ax.plot(steps, smoothed, color=color, linewidth=1.4,
                    label=f"{e['split']}  ({e['n_labeled']} lbl)")

        ax.legend(fontsize=7.5, loc='upper right')

    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({
        'font.family':        'DejaVu Sans',
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'figure.dpi':         150,
    })

    print('Discovering experiments...')
    exps = discover_experiments()
    print(f'Found {len(exps)} runs:')
    for e in exps:
        print(f"  {e['dataset']:12s}  {e['backbone']:14s}  {e['split']:6s}  "
              f"({e['n_labeled']} labeled)")

    paper_results = load_paper_results()
    print(f'Loaded paper results for: {list(paper_results.keys())}')

    figs = [
        ('1_miou_vs_labeled', fig_miou_vs_labeled(exps, paper_results)),
        ('2_training_curves', fig_training_curves(exps)),
        ('3_per_class_iou',   fig_per_class_iou(exps)),
        ('4_loss_curves',     fig_loss_curves(exps)),
    ]

    print('\nSaving figures...')
    for name, fig in figs:
        if fig is None:
            print(f'  {name}: skipped (no data)')
            continue
        fig.savefig(os.path.join(FIG_DIR, f'{name}.png'), bbox_inches='tight')
        plt.close(fig)
        print(f'  {name}: saved')

    print(f'\nDone. Figures in {FIG_DIR}/')


if __name__ == '__main__':
    main()
