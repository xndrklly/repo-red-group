#!/usr/bin/env python3
"""
Generate static spring-grid datasets for the standard ICNN case names:

  {n}_1N, {n}_10N, {n}_1N_Duffing, {n}_10N_Duffing

under data/static/ (folder names match spring_grid_static default_output_subdir).

Usage (from repo root or this directory):
  python generate_static_cases.py
  python generate_static_cases.py --grid-size 100 --seed 0 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description='Batch-generate static lattice .npz cases.')
    p.add_argument('--grid-size', type=int, default=100)
    p.add_argument('--dx', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--duffing-gamma', type=float, default=15.0)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    script = here / 'spring_grid_static.py'
    if not script.is_file():
        print(f'ERROR: missing {script}', file=sys.stderr)
        return 1

    n = int(args.grid_size)
    runs = [
        (1.0, 'linear'),
        (10.0, 'linear'),
        (1.0, 'duffing'),
        (10.0, 'duffing'),
    ]

    for force, model in runs:
        cmd = [
            sys.executable,
            str(script),
            '--grid-size',
            str(n),
            '--force',
            str(force),
            '--dx',
            str(args.dx),
            '--seed',
            str(args.seed),
            '--spring-model',
            model,
        ]
        if model == 'duffing':
            cmd += ['--duffing-gamma', str(args.duffing_gamma)]

        print('---', ' '.join(cmd), '---')
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=str(here))
        if r.returncode != 0:
            print(f'FAILED force={force} model={model}', file=sys.stderr)
            return r.returncode

    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
