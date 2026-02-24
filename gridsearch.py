"""
Successive Halving Grid Search for Inverted Residual expand_ratio.

Strategy (Successive Halving / SHA):
  - Start training ALL candidate expand_ratios for a small epoch budget.
  - Evaluate each on a held-out validation set (split from WIDER FACE).
  - Eliminate the worst half of candidates.
  - Double the epoch budget and continue training the survivors.
  - Repeat until one candidate remains → that is the optimal expand_ratio.

This is far more efficient than training every config for 300 epochs because
poor configurations are killed early while promising ones get more budget.

Usage:
    python gridsearch.py                          # defaults
    python gridsearch.py --candidates 2 3 4 6 8   # custom expand ratios
    python gridsearch.py --initial_epochs 5 --eta 3  # start at 5 epochs, prune to 1/3 each round

Reference: Jamieson & Talwalkar, "Non-stochastic Best Arm Identification and
           Hyperparameter Optimization", AISTATS 2016.
"""

from __future__ import print_function
import os
import sys
import copy
import json
import math
import time
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np

from data import AnnotationTransform, VOCDetection, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.faceboxes import FaceBoxes

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Successive Halving grid search for IR expand_ratio')

# Search space
parser.add_argument('--candidates', nargs='+', type=int, default=[2, 3, 4, 6, 8],
                    help='List of expand_ratio values to evaluate')

# Successive Halving parameters
parser.add_argument('--initial_epochs', type=int, default=5,
                    help='Epochs to train each candidate in the first round')
parser.add_argument('--eta', type=int, default=2,
                    help='Reduction factor: keep top 1/eta candidates each round '
                         '(2 = halving, 3 = keep top third)')

# Training hyper-parameters (same defaults as train.py)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--ngpu', type=int, default=1)

# Validation split
parser.add_argument('--val_fraction', type=float, default=0.1,
                    help='Fraction of training data to hold out for validation')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducible train/val split')

# Output
parser.add_argument('--training_dataset', default='./data/WIDER_FACE')
parser.add_argument('--save_folder', default='./weights/gridsearch/',
                    help='Directory to save per-candidate checkpoints')
parser.add_argument('--results_file', default='gridsearch_results.json',
                    help='JSON file to log all results')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_DIM = 1024
RGB_MEAN = (104, 117, 123)
NUM_CLASSES = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_datasets(training_dataset, val_fraction, seed):
    """Split WIDER FACE into train / val subsets."""
    full_dataset = VOCDetection(
        training_dataset, preproc(IMG_DIM, RGB_MEAN), AnnotationTransform())

    n = len(full_dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = data.random_split(full_dataset, [n_train, n_val],
                                           generator=generator)
    print(f'Dataset split: {n_train} train / {n_val} val  (total {n})')
    return train_set, val_set


def build_model(expand_ratio, device, num_gpu):
    """Create a fresh FaceBoxes model with the given expand_ratio."""
    net = FaceBoxes('train', IMG_DIM, NUM_CLASSES, expand_ratio=expand_ratio)
    if num_gpu > 1 and cfg['gpu_train']:
        net = nn.DataParallel(net, device_ids=list(range(num_gpu)))
    net = net.to(device)
    return net


def build_optimizer(net, lr, momentum, weight_decay):
    return optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                     weight_decay=weight_decay)


def adjust_learning_rate(optimizer, initial_lr, gamma, epoch, step_index,
                         iteration, epoch_size):
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** step_index)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def train_epochs(net, optimizer, criterion, priors, train_loader, device,
                 start_epoch, end_epoch, initial_lr, gamma):
    """Train from start_epoch to end_epoch (exclusive). Returns avg training
    loss over the last epoch so we can track convergence."""
    net.train()
    epoch_size = len(train_loader)

    # LR step boundaries (same as original train.py: 200, 250)
    step_values = [200, 250]

    total_loss_last_epoch = 0.0
    n_batches_last_epoch = 0

    for epoch in range(start_epoch, end_epoch):
        step_index = sum(1 for sv in step_values if epoch >= sv)

        for iteration, (images, targets) in enumerate(train_loader):
            global_iter = epoch * epoch_size + iteration

            lr = adjust_learning_rate(
                optimizer, initial_lr, gamma, epoch, step_index,
                global_iter, epoch_size)

            images = images.to(device)
            targets = [t.to(device) for t in targets]

            out = net(images)
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, priors, targets)
            loss = cfg['loc_weight'] * loss_l + loss_c
            loss.backward()
            optimizer.step()

            # Accumulate loss for the last epoch only
            if epoch == end_epoch - 1:
                total_loss_last_epoch += loss.item()
                n_batches_last_epoch += 1

        print(f'  [expand_ratio training] epoch {epoch + 1}/{end_epoch} '
              f'lr={lr:.6f}')

    avg_loss = (total_loss_last_epoch / max(n_batches_last_epoch, 1))
    return avg_loss


@torch.no_grad()
def evaluate(net, criterion, priors, val_loader, device):
    """Compute average validation loss (loc + conf)."""
    net.eval()
    total_loss = 0.0
    n = 0
    for images, targets in val_loader:
        images = images.to(device)
        targets = [t.to(device) for t in targets]
        out = net(images)
        loss_l, loss_c = criterion(out, priors, targets)
        total_loss += (cfg['loc_weight'] * loss_l + loss_c).item()
        n += 1
    net.train()
    return total_loss / max(n, 1)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Main: Successive Halving
# ---------------------------------------------------------------------------

def successive_halving(candidates, initial_epochs, eta, train_set, val_set,
                       device, num_gpu, args):
    """
    Successive Halving Algorithm.

    Each round:
      1. Train surviving candidates for `budget` more epochs.
      2. Evaluate on validation set.
      3. Keep top ceil(|survivors| / eta) candidates.
      4. Double the budget.
    """
    os.makedirs(args.save_folder, exist_ok=True)

    # Prior boxes (shared across all models — same architecture output sizes)
    priorbox = PriorBox(cfg, image_size=(IMG_DIM, IMG_DIM))
    priors = priorbox.forward().to(device)

    criterion = MultiBoxLoss(NUM_CLASSES, 0.35, True, 0, True, 7, 0.35, False)

    # Data loaders
    train_loader = data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=detection_collate,
        drop_last=True)
    val_loader = data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate,
        drop_last=False)

    # State for each candidate: model, optimizer, epoch counter
    survivors = {}
    for er in candidates:
        net = build_model(er, device, num_gpu)
        opt = build_optimizer(net, args.lr, args.momentum, args.weight_decay)
        n_params = count_parameters(net)
        survivors[er] = {
            'net': net,
            'optimizer': opt,
            'epoch': 0,
            'val_loss': float('inf'),
            'params': n_params,
        }
        print(f'  Candidate expand_ratio={er}  params={n_params:,}')

    budget = initial_epochs  # epochs to train in this round
    round_num = 0
    log = []  # list of dicts persisted to JSON

    # Calculate total number of rounds
    n_rounds = max(1, math.ceil(math.log(len(candidates)) / math.log(eta)))
    total_epochs_upper = budget * (2 ** n_rounds)
    print(f'\n=== Successive Halving: {len(candidates)} candidates, '
          f'eta={eta}, initial_budget={budget}, '
          f'~{n_rounds} rounds, max ~{total_epochs_upper} epochs for the winner ===\n')

    while len(survivors) > 1:
        round_num += 1
        print(f'\n--- Round {round_num}: training {len(survivors)} survivors '
              f'for {budget} epochs (total so far → '
              f'{list(survivors.values())[0]["epoch"] + budget} epochs) ---')

        for er, state in survivors.items():
            start_ep = state['epoch']
            end_ep = start_ep + budget

            print(f'\n  ▸ expand_ratio={er}  training epochs [{start_ep+1}..{end_ep}]')
            t0 = time.time()
            train_loss = train_epochs(
                state['net'], state['optimizer'], criterion, priors,
                train_loader, device, start_ep, end_ep,
                args.lr, args.gamma)
            train_time = time.time() - t0

            val_loss = evaluate(state['net'], criterion, priors, val_loader, device)
            state['epoch'] = end_ep
            state['val_loss'] = val_loss

            print(f'    train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
                  f'time={datetime.timedelta(seconds=int(train_time))}')

            log.append({
                'round': round_num,
                'expand_ratio': er,
                'epochs_trained': end_ep,
                'train_loss': round(train_loss, 5),
                'val_loss': round(val_loss, 5),
                'params': state['params'],
                'train_seconds': round(train_time, 1),
            })

            # Save checkpoint
            ckpt_path = os.path.join(
                args.save_folder, f'er{er}_epoch{end_ep}.pth')
            torch.save(state['net'].state_dict(), ckpt_path)

        # Rank by val_loss (lower is better) and keep top ceil(n/eta)
        ranked = sorted(survivors.items(), key=lambda kv: kv[1]['val_loss'])
        n_keep = max(1, math.ceil(len(ranked) / eta))
        kept = [er for er, _ in ranked[:n_keep]]
        pruned = [er for er, _ in ranked[n_keep:]]

        print(f'\n  Ranking (val_loss): '
              + '  '.join(f'er={er} → {s["val_loss"]:.4f}'
                          for er, s in ranked))
        print(f'  ✓ Keeping:  {kept}')
        if pruned:
            print(f'  ✗ Pruning:  {pruned}')

        # Free pruned models to reclaim GPU memory
        for er in pruned:
            del survivors[er]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Double budget for next round
        budget *= eta

    # --- Winner ---
    winner_er, winner_state = list(survivors.items())[0]
    print(f'\n{"="*60}')
    print(f'  WINNER: expand_ratio = {winner_er}')
    print(f'  Trained for {winner_state["epoch"]} epochs')
    print(f'  Val loss: {winner_state["val_loss"]:.4f}')
    print(f'  Params: {winner_state["params"]:,}')
    print(f'{"="*60}\n')

    # Save winner
    final_path = os.path.join(args.save_folder, f'BEST_er{winner_er}.pth')
    torch.save(winner_state['net'].state_dict(), final_path)
    print(f'Saved best model → {final_path}')

    # Persist full log
    results = {
        'winner': {
            'expand_ratio': winner_er,
            'epochs_trained': winner_state['epoch'],
            'val_loss': winner_state['val_loss'],
            'params': winner_state['params'],
        },
        'candidates': list(candidates),
        'settings': {
            'initial_epochs': args.initial_epochs,
            'eta': args.eta,
            'val_fraction': args.val_fraction,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'lr': args.lr,
        },
        'rounds': log,
    }
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Full results → {args.results_file}')

    return winner_er, results


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    start_wall = time.perf_counter()

    cudnn.benchmark = True
    gpu_train = cfg['gpu_train']
    device = torch.device('cuda:0' if gpu_train and torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}')

    train_set, val_set = make_datasets(
        args.training_dataset, args.val_fraction, args.seed)

    winner, results = successive_halving(
        candidates=args.candidates,
        initial_epochs=args.initial_epochs,
        eta=args.eta,
        train_set=train_set,
        val_set=val_set,
        device=device,
        num_gpu=args.ngpu,
        args=args,
    )

    elapsed = time.perf_counter() - start_wall
    print(f'\nTotal wall time: {datetime.timedelta(seconds=int(elapsed))}')
