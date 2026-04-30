"""
Verification tests for block-averaged spring-grid reduced-order helpers.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import sindy_torch
import spring_grid_K_regression as base
from sindy_torch.systems import get_free_dofs


PASS_COUNT = 0
FAIL_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}  {detail}")


def test_full_grid_reduction():
    print("\n=== Test 1: Full-grid block averaging ===")
    reduction = sindy_torch.build_block_average_reduction(6, 3)
    check("coarse grid size", reduction.coarse_grid_size == 2)
    check("restriction shape", reduction.restriction.shape == (4, 36))
    check("prolongation shape", reduction.prolongation.shape == (36, 4))
    check(
        "R @ P = I",
        np.allclose(
            reduction.restriction @ reduction.prolongation,
            np.eye(reduction.coarse_grid_size ** 2),
        ),
    )

    snapshot = np.arange(36, dtype=float).reshape(6, 6)
    reduced = sindy_torch.apply_block_average_reduction(snapshot.ravel(), reduction)
    expected = np.array([
        snapshot[:3, :3].mean(),
        snapshot[:3, 3:].mean(),
        snapshot[3:, :3].mean(),
        snapshot[3:, 3:].mean(),
    ])
    check("block averages match expected means", np.allclose(reduced, expected))


def test_free_dof_reduction():
    print("\n=== Test 2: Free-DOF-aware block averaging ===")
    _, free = get_free_dofs(10, pinned_rows=(0,))
    reduction = sindy_torch.build_block_average_reduction(10, 2, free_dofs=free)
    top_row_counts = reduction.active_counts[:5]
    lower_counts = reduction.active_counts[5:]
    check(
        "support-touching blocks average active nodes only",
        np.array_equal(top_row_counts, np.full(5, 2)),
        f"counts={top_row_counts.tolist()}",
    )
    check(
        "interior blocks keep 4 active nodes",
        np.array_equal(lower_counts, np.full(20, 4)),
        f"counts={lower_counts.tolist()}",
    )


def test_reduced_dataset_and_regression(device: torch.device):
    print("\n=== Test 3: Reduced dataset feeds regression path ===")
    data = base.generate_dataset(
        n=4,
        n_traj=3,
        t_end=0.3,
        dt=0.02,
        seed=0,
        device=torch.device("cpu"),
        simulation_backend="scipy",
        dtype=torch.float64,
    )
    rom_data = base.build_block_average_rom_dataset(data, block_size=2)
    rom_problem = base.build_regression_problem(
        rom_data,
        device=device,
        dtype=torch.float64,
        state_key="U_rom",
        target_key="A_target_rom",
        stiffness_key="K_rom",
        mask_key="locality_rom",
    )
    check("reduced coarse grid size", rom_data["n_rom"] == 2)
    check("reduced state width", rom_data["U_rom"].shape[1] == 4)
    check("reduced trajectory fields exist", "U_rom" in rom_data["trajectories"][0])
    check("reduced regression theta width", rom_problem["theta_train"].shape[1] == 4)
    check("reduced stiffness shape", tuple(rom_problem["K_true"].shape) == (4, 4))
    check(
        "reduced trajectory initial state matches first sample",
        np.allclose(
            rom_data["trajectories"][0]["u0_rom"],
            rom_data["trajectories"][0]["U_rom"][0],
        ),
    )


def test_invalid_block_size():
    print("\n=== Test 4: Invalid block size is rejected ===")
    try:
        sindy_torch.build_block_average_reduction(10, 3)
    except ValueError as exc:
        check(
            "non-divisible grid size raises",
            "integer multiple" in str(exc),
            str(exc),
        )
    else:
        check("non-divisible grid size raises", False, "expected ValueError")


def main(device_arg="cpu"):
    device = sindy_torch.get_device(device_arg)
    print(f"Device: {device}")
    test_full_grid_reduction()
    test_free_dof_reduction()
    test_reduced_dataset_and_regression(device)
    test_invalid_block_size()

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("All block-averaged spring-grid ROM verifications PASSED.")
    else:
        print("Some block-averaged spring-grid ROM verifications FAILED; see above.")
    print(f"{'=' * 50}")
    return FAIL_COUNT == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    success = main(args.device)
    sys.exit(0 if success else 1)
