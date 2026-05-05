#!/usr/bin/env python3
"""
Train the ICNN pipeline (run.py) on several prepared static datasets.

Expects each case folder under <repo>/data/static/<case>/lattice_static.npz
(e.g. names from spring_grid_static: 100_1N, 100_10N, 100_1N_Duffing, ...).
Generate those with:  data_generation/generate_static_cases.py (or .bat).

Usage (from repo root or PDE/ICNN):
  python train_static_cases.py
  python train_static_cases.py --cases 100_1N 100_10N --dry-run
  python train_static_cases.py -- python run.py extra args not used here
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CASES = [
    '100_1N',
    '100_10N',
    '100_1N_Duffing',
    '100_10N_Duffing',
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run ICNN training (run.py --force-train) on static case folders.',
    )
    parser.add_argument(
        '--cases',
        nargs='*',
        default=DEFAULT_CASES,
        help=f'Folder names under data/static/ (default: {DEFAULT_CASES})',
    )
    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='Skip cases where lattice_static.npz is missing',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands only',
    )
    parser.add_argument(
        'run_py_args',
        nargs=argparse.REMAINDER,
        help='Extra args passed to run.py after -- (e.g. -- --n-ensemble 3)',
    )
    args = parser.parse_args()
    extra = args.run_py_args
    if extra and extra[0] == '--':
        extra = extra[1:]

    here = Path(__file__).resolve()
    project_root = here.parents[2]
    run_py = here.parent / 'run.py'
    data_root = project_root / 'data' / 'static'

    if not run_py.is_file():
        print(f'ERROR: run.py not found at {run_py}', file=sys.stderr)
        return 1

    ok = 0
    for case in args.cases:
        npz = data_root / case / 'lattice_static.npz'
        if not npz.is_file():
            msg = f'Missing: {npz}'
            if args.skip_missing:
                print(f'SKIP  {msg}')
                continue
            print(f'ERROR {msg}', file=sys.stderr)
            return 1

        cmd = [
            sys.executable,
            str(run_py),
            '--data',
            str(npz),
            '--force-train',
            *extra,
        ]
        print(f'--- {case} ---')
        print(' ', ' '.join(cmd))
        if args.dry_run:
            ok += 1
            continue
        r = subprocess.run(cmd, cwd=str(run_py.parent))
        if r.returncode != 0:
            print(f'FAILED {case} (exit {r.returncode})', file=sys.stderr)
            return r.returncode
        ok += 1

    print(f'Done. Trained {ok} case(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
