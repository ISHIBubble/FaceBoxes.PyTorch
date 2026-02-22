"""
Successive Halving Grid Search for Focal Loss + CIoU hyperparameters.

Efficiently finds optimal (focal_alpha, focal_gamma) by:
  1. Training ALL combos for a short budget (e.g. 15 epochs)
  2. Keeping only the top-performing half
  3. Continuing to train survivors for another budget round
  4. Repeating until 1-2 configs remain

This is ~10x more efficient than exhaustive grid search to max_epoch.

Usage:
    python grid_search.py                          # defaults
    python grid_search.py --epochs_per_round 20    # longer rounds
    python grid_search.py --resume results/grid_search_results.json
"""

from __future__ import print_function
import os
import sys
import json
import copy
import time
import math
import datetime
import itertools
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import FocalCIoULoss
from layers.functions.prior_box import PriorBox
from models.faceboxes import FaceBoxes


# ───────────────────────── CLI ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Successive Halving Grid Search')

    # Search space
    p.add_argument('--alphas', nargs='+', type=float,
                   default=[0.10, 0.25, 0.40, 0.55, 0.75],
                   help='Focal Loss alpha values to search')
    p.add_argument('--gammas', nargs='+', type=float,
                   default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                   help='Focal Loss gamma values to search')

    # Successive halving
    p.add_argument('--epochs_per_round', default=15, type=int,
                   help='Epochs to train each surviving config per round')
    p.add_argument('--min_survivors', default=1, type=int,
                   help='Stop halving when this many configs remain')
    p.add_argument('--keep_ratio', default=0.5, type=float,
                   help='Fraction of configs to keep each round (default 0.5)')

    # Training
    p.add_argument('--training_dataset', default='./data/WIDER_FACE')
    p.add_argument('-b', '--batch_size', default=32, type=int)
    p.add_argument('--num_workers', default=16, type=int)
    p.add_argument('--ngpu', default=1, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--weight_decay', default=5e-4, type=float)
    p.add_argument('--gamma', default=0.1, type=float, help='LR step decay factor')

    # Output
    p.add_argument('--results_dir', default='./results',
                   help='Directory to save results and checkpoints')
    p.add_argument('--resume', default=None, type=str,
                   help='Path to grid_search_results.json to resume from')

    return p.parse_args()


# ───────────── Training helpers ─────────────
IMG_DIM = 1024
RGB_MEAN = (104, 117, 123)
NUM_CLASSES = 2


def build_model(device, ngpu):
    """Create a fresh FaceBoxes model on the given device."""
    net = FaceBoxes('train', IMG_DIM, NUM_CLASSES)
    if ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))
    net = net.to(device)
    return net


def build_priors(device):
    """Generate prior boxes (anchors)."""
    priorbox = PriorBox(cfg, image_size=(IMG_DIM, IMG_DIM))
    with torch.no_grad():
        priors = priorbox.forward().to(device)
    return priors


def train_config(net, optimizer, criterion, priors, dataset, device, args,
                 start_epoch, end_epoch):
    """
    Train *net* from start_epoch to end_epoch.

    Returns:
        avg_loss: average total loss over the last epoch
        avg_loss_l: average localization loss over the last epoch
        avg_loss_c: average classification loss over the last epoch
    """
    net.train()
    batch_size = args.batch_size
    epoch_size = math.ceil(len(dataset) / batch_size)
    loc_weight = cfg['loc_weight']

    # LR step schedule (same as baseline): step at epoch 200 and 250
    # Since grid search runs for fewer epochs, most configs won't hit these.
    step_epochs = [200, 250]

    avg_loss = avg_loss_l = avg_loss_c = 0.0

    for epoch in range(start_epoch, end_epoch):
        # --- set learning rate based on epoch ---
        step_index = sum(1 for s in step_epochs if epoch >= s)
        lr = args.lr * (args.gamma ** step_index)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        batch_iter = iter(data.DataLoader(
            dataset, batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=detection_collate))

        epoch_loss = epoch_loss_l = epoch_loss_c = 0.0
        n_batches = 0

        for iteration in range(epoch_size):
            try:
                images, targets = next(batch_iter)
            except StopIteration:
                break

            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)
            loss = loc_weight * loss_l + loss_c

            # Detect NaN / Inf loss — abort this config early
            if not torch.isfinite(loss):
                print(f'    *** NaN/Inf loss detected at epoch {epoch + 1}, '
                      f'iteration {iteration}. Aborting config. ***')
                return float('nan'), float('nan'), float('nan')

            loss.backward()
            # Gradient clipping to prevent explosion with low gamma
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_l += loss_l.item()
            epoch_loss_c += loss_c.item()
            n_batches += 1

        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_loss_l = epoch_loss_l / n_batches
            avg_loss_c = epoch_loss_c / n_batches

        print(f'    Epoch {epoch + 1}/{end_epoch}  '
              f'L: {avg_loss_l:.4f}  C: {avg_loss_c:.4f}  '
              f'Total: {avg_loss:.4f}  LR: {lr:.6f}')

    return avg_loss, avg_loss_l, avg_loss_c


# ────────────── Successive Halving ──────────────
def successive_halving(args):
    device = torch.device('cuda:0' if cfg['gpu_train'] else 'cpu')
    cudnn.benchmark = True

    priors = build_priors(device)

    # Dataset (loaded once, shared across all configs)
    print('Loading Dataset...')
    dataset = VOCDetection(args.training_dataset,
                           preproc(IMG_DIM, RGB_MEAN), AnnotationTransform())
    print(f'Dataset size: {len(dataset)} images')

    # Build the (alpha, gamma) grid
    all_combos = list(itertools.product(args.alphas, args.gammas))
    print(f'\nGrid: {len(args.alphas)} alphas × {len(args.gammas)} gammas = '
          f'{len(all_combos)} configurations')

    # ---- Resume support ----
    if args.resume and os.path.isfile(args.resume):
        print(f'\nResuming from {args.resume}')
        with open(args.resume, 'r') as f:
            saved = json.load(f)
        configs = saved['configs']
        completed_rounds = saved.get('completed_rounds', 0)
        # rebuild combo list from saved active configs
        active_indices = [i for i, c in enumerate(configs) if c['active']]
        print(f'  {len(active_indices)} active configs, '
              f'{completed_rounds} rounds already done')
    else:
        configs = []
        for i, (alpha, gamma) in enumerate(all_combos):
            configs.append({
                'id': i,
                'focal_alpha': alpha,
                'focal_gamma': gamma,
                'epochs_trained': 0,
                'loss_history': [],      # total loss after each round
                'loss_l_history': [],
                'loss_c_history': [],
                'active': True,
                'checkpoint': None,      # path to saved state_dict
            })
        completed_rounds = 0

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'checkpoints'), exist_ok=True)

    round_num = completed_rounds
    while True:
        active = [c for c in configs if c['active']]
        if len(active) <= args.min_survivors:
            break

        round_num += 1
        start_epoch_global = active[0]['epochs_trained']
        end_epoch_global = start_epoch_global + args.epochs_per_round

        print(f'\n{"=" * 70}')
        print(f'ROUND {round_num}: training {len(active)} configs  '
              f'epochs {start_epoch_global + 1}→{end_epoch_global}')
        print(f'{"=" * 70}')

        for cfg_entry in active:
            alpha = cfg_entry['focal_alpha']
            gamma = cfg_entry['focal_gamma']
            cid = cfg_entry['id']

            print(f'\n  Config {cid}: alpha={alpha}, gamma={gamma}')

            # Build fresh model (or load checkpoint)
            net = build_model(device, args.ngpu)
            optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)

            # Restore from checkpoint if available
            if cfg_entry['checkpoint'] and os.path.isfile(cfg_entry['checkpoint']):
                ckpt = torch.load(cfg_entry['checkpoint'], map_location=device)
                net.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt['optimizer'])
                print(f'    Resumed from {cfg_entry["checkpoint"]}')

            criterion = FocalCIoULoss(
                num_classes=NUM_CLASSES,
                overlap_thresh=0.35,
                prior_for_matching=True,
                bkg_label=0,
                neg_mining=True,
                neg_pos=7,
                neg_overlap=0.35,
                encode_target=False,
                focal_alpha=alpha,
                focal_gamma=gamma,
            )

            t0 = time.perf_counter()
            avg_loss, avg_loss_l, avg_loss_c = train_config(
                net, optimizer, criterion, priors, dataset, device, args,
                start_epoch=cfg_entry['epochs_trained'],
                end_epoch=end_epoch_global,
            )
            elapsed = time.perf_counter() - t0

            # Save checkpoint
            ckpt_path = os.path.join(
                args.results_dir, 'checkpoints', f'config_{cid}.pth')
            torch.save({
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)

            cfg_entry['epochs_trained'] = end_epoch_global
            cfg_entry['loss_history'].append(avg_loss)
            cfg_entry['loss_l_history'].append(avg_loss_l)
            cfg_entry['loss_c_history'].append(avg_loss_c)
            cfg_entry['checkpoint'] = ckpt_path

            # Immediately eliminate configs that produced NaN
            if math.isnan(avg_loss):
                cfg_entry['active'] = False
                print(f'    ELIMINATED (NaN loss) in {elapsed:.1f}s')
                if os.path.isfile(ckpt_path):
                    os.remove(ckpt_path)
                    cfg_entry['checkpoint'] = None
            else:
                print(f'    Done in {elapsed:.1f}s  '
                      f'Loss: {avg_loss:.4f} (L:{avg_loss_l:.4f} C:{avg_loss_c:.4f})')

            # Free GPU memory between configs
            del net, optimizer, criterion
            torch.cuda.empty_cache()

        # ---- Halving: keep top fraction ----
        active = [c for c in configs if c['active']]

        def sort_key(c):
            """NaN-safe sort key: NaN → infinity so it always ranks last."""
            val = c['loss_history'][-1] if c['loss_history'] else float('inf')
            return val if not math.isnan(val) else float('inf')

        # Sort by last observed total loss (lower is better)
        active.sort(key=sort_key)

        n_keep = max(args.min_survivors,
                     int(math.ceil(len(active) * args.keep_ratio)))
        eliminated = active[n_keep:]

        for c in eliminated:
            c['active'] = False
            # Remove checkpoint to save disk
            if c['checkpoint'] and os.path.isfile(c['checkpoint']):
                os.remove(c['checkpoint'])
                c['checkpoint'] = None

        # Save intermediate results
        results_path = os.path.join(args.results_dir, 'grid_search_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'completed_rounds': round_num,
                'configs': configs,
                'args': vars(args),
            }, f, indent=2)

        print(f'\n--- Round {round_num} results ---')
        print(f'{"ID":>4} {"Alpha":>7} {"Gamma":>7} {"Loss":>10} {"Status":>10}')
        for c in sorted(configs, key=sort_key):
            last_loss = c['loss_history'][-1] if c['loss_history'] else float('inf')
            loss_str = f'{last_loss:10.4f}' if not math.isnan(last_loss) else '       NaN'
            status = 'ACTIVE' if c['active'] else 'eliminated'
            print(f'{c["id"]:4d} {c["focal_alpha"]:7.2f} {c["focal_gamma"]:7.2f} '
                  f'{loss_str} {status:>10}')

        surviving = [c for c in configs if c['active']]
        print(f'\nSurvivors: {len(surviving)} / {len(configs)}')

    # ──── Final results ────
    final_active = [c for c in configs if c['active']]
    final_active.sort(key=lambda c: c['loss_history'][-1])
    best = final_active[0]

    print(f'\n{"=" * 70}')
    print(f'GRID SEARCH COMPLETE')
    print(f'{"=" * 70}')
    print(f'Best config:  alpha={best["focal_alpha"]}, gamma={best["focal_gamma"]}')
    print(f'  Final loss: {best["loss_history"][-1]:.4f}  '
          f'(L: {best["loss_l_history"][-1]:.4f}, C: {best["loss_c_history"][-1]:.4f})')
    print(f'  Trained for {best["epochs_trained"]} epochs')
    print(f'  Checkpoint: {best["checkpoint"]}')
    print(f'\nFull results saved to: {os.path.join(args.results_dir, "grid_search_results.json")}')

    # Save winning checkpoint with a descriptive name
    if best['checkpoint'] and os.path.isfile(best['checkpoint']):
        winner_name = (f'FaceBoxes_FL_alpha{best["focal_alpha"]}'
                       f'_gamma{best["focal_gamma"]}.pth')
        winner_path = os.path.join(args.results_dir, winner_name)
        ckpt = torch.load(best['checkpoint'], map_location='cpu')
        torch.save(ckpt['model'], winner_path)
        print(f'  Winner weights saved to: {winner_path}')

    return best


if __name__ == '__main__':
    args = parse_args()
    t0 = time.perf_counter()
    best = successive_halving(args)
    elapsed = time.perf_counter() - t0
    print(f'\nTotal grid search time: {datetime.timedelta(seconds=int(elapsed))}')
