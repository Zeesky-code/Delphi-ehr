"""
Score trained runs on a data split from their saved checkpoints, separately from training, so the
test split is only touched when we decide to (the abstract's numbers come from here).

For each run directory, loads ckpt_last.pt ('final' checkpoint rule) or ckpt_best.pt, scores every
patient of the split with utils.evaluate_split (fixed order, regular padding; CE overall and on the
lab-linked diseases, per group) and writes <run>/eval_<split>_<ckpt>.json.
compare_runs.py --split test reads these files.

Usage:
    python3 evaluate.py runs/film-mimic4/*/seed* --split test [--ckpt last|best] [--device cuda]
"""

import argparse
import json
import os

import numpy as np
import torch

from model import Delphi, DelphiConfig
from utils import evaluate_split, get_p2i, shuffle_values_within_tokens

CKPT_FILES = {'last': ['ckpt_last.pt', 'ckpt.pt'], 'best': ['ckpt_best.pt', 'ckpt.pt']}


def load_run(run_dir, ckpt='last', device='cpu'):
    """Model (eval mode) and checkpoint dict of a run. Older runs only have ckpt.pt (their best)."""
    path = next((os.path.join(run_dir, f) for f in CKPT_FILES[ckpt]
                 if os.path.exists(os.path.join(run_dir, f))), None)
    if path is None:
        raise FileNotFoundError(f'no {ckpt} checkpoint in {run_dir}')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Delphi(DelphiConfig(**checkpoint['model_args']))
    model.load_state_dict({k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v
                           for k, v in checkpoint['model'].items()})
    model.to(device).eval()
    checkpoint['path'] = path
    return model, checkpoint


def load_split(data_dir, split, checkpoint):
    """
    Events of a split and, for FiLM models, its lab values: shuffled exactly as in training for the
    shuffled-values control, and normalised later with the statistics saved from the training split.
    """
    data = np.fromfile(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    values = None
    if checkpoint['model_args'].get('use_film') and checkpoint.get('values_stats') is not None:
        values = np.fromfile(os.path.join(data_dir, f'{split}_values.bin'), dtype=np.float32)
        if checkpoint.get('config', {}).get('film_shuffle_values'):
            values = shuffle_values_within_tokens(data, values)
    return data, values


def data_dir_of(checkpoint, override=None):
    return override or os.path.join('data', checkpoint['config']['dataset'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='run directories (containing ckpt_last.pt / ckpt_best.pt)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--ckpt', default='last', choices=['last', 'best'],
                        help="last = the 'final' checkpoint rule (default), best = lowest periodic val loss")
    parser.add_argument('--data', help='dataset directory (default: data/<dataset of the run>)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    for run_dir in args.runs:
        model, checkpoint = load_run(run_dir, args.ckpt, args.device)
        data_dir = data_dir_of(checkpoint, args.data)
        data, values = load_split(data_dir, args.split, checkpoint)
        linked_path = os.path.join(data_dir, 'lab_linked_tokens.json')
        groups = json.load(open(linked_path))['groups'] if os.path.exists(linked_path) else {}
        conf = checkpoint['config']
        result = evaluate_split(model, data, get_p2i(data), values, checkpoint.get('values_stats'), groups,
                                checkpoint['model_args']['ignore_tokens'], checkpoint['model_args']['block_size'],
                                args.batch_size, args.device, conf.get('no_event_token_rate', 5))
        result = {'split': args.split, 'ckpt': os.path.basename(checkpoint['path']),
                  'iter_num': checkpoint['iter_num'], 'seed': conf.get('seed'), **result}
        out = os.path.join(run_dir, f'eval_{args.split}_{args.ckpt}.json')
        with open(out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"{run_dir}: loss {result['loss']:.4f}, CE {result['ce']:.4f}, "
              f"CE linked {result.get('ce_linked') or float('nan'):.4f} -> {out}")


if __name__ == '__main__':
    main()
