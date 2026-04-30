"""
Spring-grid simulation checks for the torch Newmark-beta backend.

These tests always verify that the new torch implementation matches the
existing scipy reference on CPU, and run an extra CUDA consistency check when
CUDA is available in the current PyTorch build.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sindy_torch.systems import (
    build_stiffness_matrix,
    get_free_dofs,
    newmark_beta_simulation,
    newmark_beta_simulation_torch,
    zero_force,
    zero_force_torch,
)


PASS_COUNT = 0
FAIL_COUNT = 0
SKIP_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}  {detail}")


def skip(name, detail="CUDA unavailable"):
    global SKIP_COUNT
    SKIP_COUNT += 1
    print(f"  SKIP: {name}  {detail}")


def max_abs_diff(a, b) -> float:
    a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else np.asarray(b)
    return float(np.max(np.abs(a_np - b_np)))


def build_problem():
    n = 4
    N = n * n
    K = build_stiffness_matrix(n, seed=7, k_low=0.5, k_high=1.5)
    M = sp.eye(N, format="csr")
    C = sp.csr_matrix((N, N))
    _, free = get_free_dofs(n, pinned_rows=(0,))

    rng = np.random.default_rng(19)
    u0 = np.zeros(N)
    v0 = np.zeros(N)
    u0[free] = 0.15 * rng.standard_normal(free.size)
    v0[free] = 0.10 * rng.standard_normal(free.size)
    return n, N, K, M, C, free, u0, v0


def test_scipy_vs_torch_cpu():
    print("\n=== Test 1: scipy vs torch CPU spring-grid simulation ===")
    n, N, K, M, C, free, u0, v0 = build_problem()
    result_ref = newmark_beta_simulation(
        K,
        M,
        C,
        zero_force(N),
        free,
        n,
        t_end=0.2,
        dt=0.01,
        u0=u0,
        v0=v0,
    )
    result_torch = newmark_beta_simulation_torch(
        K,
        M,
        C,
        zero_force_torch(N, device="cpu", dtype=torch.float64),
        free,
        n,
        t_end=0.2,
        dt=0.01,
        u0=u0,
        v0=v0,
        device="cpu",
        dtype=torch.float64,
    )

    check("CPU times match", max_abs_diff(result_ref.times, result_torch.times) < 1e-12)
    check("CPU displacements match", max_abs_diff(result_ref.displacements, result_torch.displacements) < 1e-9)
    check("CPU velocities match", max_abs_diff(result_ref.velocities, result_torch.velocities) < 1e-9)
    check("CPU accelerations match", max_abs_diff(result_ref.accelerations, result_torch.accelerations) < 1e-8)


def test_torch_cpu_vs_cuda():
    print("\n=== Test 2: torch CPU vs torch CUDA spring-grid simulation ===")
    if not torch.cuda.is_available():
        skip("torch CPU vs torch CUDA spring-grid simulation")
        return

    n, N, K, M, C, free, u0, v0 = build_problem()
    result_cpu = newmark_beta_simulation_torch(
        K,
        M,
        C,
        zero_force_torch(N, device="cpu", dtype=torch.float64),
        free,
        n,
        t_end=0.2,
        dt=0.01,
        u0=u0,
        v0=v0,
        device="cpu",
        dtype=torch.float64,
    )
    result_gpu = newmark_beta_simulation_torch(
        K,
        M,
        C,
        zero_force_torch(N, device="cuda", dtype=torch.float64),
        free,
        n,
        t_end=0.2,
        dt=0.01,
        u0=u0,
        v0=v0,
        device="cuda",
        dtype=torch.float64,
    )

    check("CUDA displacements match CPU", max_abs_diff(result_cpu.displacements, result_gpu.displacements) < 1e-9)
    check("CUDA velocities match CPU", max_abs_diff(result_cpu.velocities, result_gpu.velocities) < 1e-9)
    check("CUDA accelerations match CPU", max_abs_diff(result_cpu.accelerations, result_gpu.accelerations) < 1e-8)


def main():
    test_scipy_vs_torch_cpu()
    test_torch_cpu_vs_cuda()

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {SKIP_COUNT} skipped")
    if FAIL_COUNT == 0:
        print("Spring-grid torch simulation checks PASSED.")
    else:
        print("Spring-grid torch simulation checks FAILED; see above.")
    print(f"{'=' * 50}")
    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
