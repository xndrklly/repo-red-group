"""
Verification tests for the spring-grid trajectory-v2 workflow.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import sindy_torch
import spring_grid_K_regression as base
import generate_spring_grid_three_model_reports as reports


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


def build_small_problem(device: torch.device, dtype: torch.dtype = torch.float64):
    data = base.generate_dataset(
        n=4,
        n_traj=4,
        t_end=0.8,
        dt=0.02,
        seed=0,
        device=torch.device("cpu"),
        simulation_backend="scipy",
        dtype=dtype,
    )
    grad_problem = base.build_regression_problem(data, device=device, dtype=dtype)
    selected_nodes = base.select_representative_nodes(
        data["n"], data["free"], data["trajectories"][0]["U_full"]
    )
    incident_specs = base.build_incident_specs(selected_nodes, data["n"])
    residual = base.train_gradient_method(
        grad_problem["theta_train"],
        grad_problem["target_train"],
        grad_problem["theta_test"],
        grad_problem["target_test"],
        grad_problem["K_true"],
        n_features=grad_problem["n_features"],
        n_states=grad_problem["n_states"],
        n_epochs=40,
        lr=5e-2,
        l1_lambda=0.0,
        proximal=False,
        snapshot_every=5,
        n_eig=5,
        seed=0,
        xi_mask=grad_problem["xi_mask"],
        history_specs=incident_specs,
        history_every=1,
        free=data["free"],
        n_total_dofs=data["N"],
    )
    legacy = reports.train_trajectory_legacy_method(
        data["trajectories"],
        grad_problem["K_true"],
        n_states=grad_problem["n_states"],
        n_epochs=25,
        lr=1e-2,
        seed=0,
        xi_mask=grad_problem["xi_mask"],
        history_specs=incident_specs,
        history_every=1,
        free=data["free"],
        n_total_dofs=data["N"],
        sample_stride=2,
        n_eig=5,
    )
    v2 = reports.train_trajectory_v2_method(
        data["trajectories"],
        grad_problem["K_true"],
        warm_start_xi=residual["xi"],
        n=data["n"],
        n_states=grad_problem["n_states"],
        n_epochs=25,
        lr=1e-3,
        seed=0,
        xi_mask=grad_problem["xi_mask"],
        history_specs=incident_specs,
        history_every=1,
        free=data["free"],
        n_total_dofs=data["N"],
        window_length=21,
        window_stride=10,
        patience=8,
        n_eig=5,
    )
    return data, grad_problem, residual, legacy, v2


def test_v2_training(device: torch.device):
    print("\n=== Test 1: Trajectory v2 improves windowed validation loss ===")
    data, grad_problem, residual, legacy, v2 = build_small_problem(device)
    check(
        "v2 validation improves from warm start",
        v2["best_test_loss"] < v2["warm_start_test_loss"],
        f"warm={v2['warm_start_test_loss']:.4e}, best={v2['best_test_loss']:.4e}",
    )
    check(
        "v2 stiffness stays symmetric",
        v2["K_symmetric_max_abs_diff"] < 1e-10,
        f"sym_diff={v2['K_symmetric_max_abs_diff']:.4e}",
    )
    rollout = base.rollout_linear_model(
        base.xi_to_K(v2["xi"]),
        data["free"],
        data["n"],
        dt=data["dt"],
        t_end=data["t_end"],
        u0_full=data["trajectories"][0]["u0_full"],
        v0_full=data["trajectories"][0]["v0_full"],
        device=torch.device("cpu"),
        simulation_backend="scipy",
        dtype=torch.float64,
    )
    check(
        "v2 rollout stays finite",
        np.isfinite(rollout["U_full"]).all() and np.isfinite(rollout["V_full"]).all(),
    )

    print("\n=== Test 2: Trajectory v2 beats legacy trajectory loss ===")
    check(
        "v2 validation beats legacy",
        v2["test_losses"][-1] < legacy["test_losses"][-1],
        f"v2={v2['test_losses'][-1]:.4e}, legacy={legacy['test_losses'][-1]:.4e}",
    )
    warm_test_deriv = base.derivative_mse(
        grad_problem["theta_test"],
        residual["xi"],
        grad_problem["target_test"],
    )
    v2_test_deriv = base.derivative_mse(
        grad_problem["theta_test"],
        v2["xi"],
        grad_problem["target_test"],
    )
    check(
        "v2 derivative mse stays near warm start",
        v2_test_deriv <= 1.30 * warm_test_deriv,
        f"warm={warm_test_deriv:.4e}, v2={v2_test_deriv:.4e}",
    )


def test_report_smoke(device: torch.device):
    print("\n=== Test 3: Report smoke path honors trajectory variants ===")
    device_arg = "cuda" if device.type == "cuda" else "cpu"
    sim_backend = "torch" if device.type == "cuda" else "scipy"
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        out_v2 = root / "figures" / "small_v2_report.html"
        reports.generate_report(
            report_title="Small V2 Report",
            spring_description="smoke test",
            out_path=out_v2,
            stiffness_matrix=None,
            dataset_seed=0,
            residual_epochs=4,
            trajectory_epochs=4,
            trajectory_stride=2,
            trajectory_variant="v2",
            trajectory_window=6,
            trajectory_window_stride=3,
            trajectory_patience=2,
            execution_mode="single",
            device_arg=device_arg,
            simulation_backend=sim_backend,
            n=4,
            n_traj=4,
            t_end=0.2,
            dt=0.02,
        )
        html_v2 = out_v2.read_text(encoding="utf-8")
        check("v2 report contains v2 section", reports.ADAM_TRAJECTORY_V2_NAME in html_v2)
        check("v2 report omits legacy section", reports.ADAM_TRAJECTORY_LEGACY_NAME not in html_v2)
        check(
            "v2 report figure dir created",
            (root / "results" / "figures" / "small_v2").exists(),
        )
        check(
            "v2 report animation dir created",
            (root / "results" / "animations" / "small_v2").exists(),
        )

        out_both = root / "figures" / "small_both_report.html"
        reports.generate_report(
            report_title="Small Both Report",
            spring_description="smoke test",
            out_path=out_both,
            stiffness_matrix=None,
            dataset_seed=0,
            residual_epochs=4,
            trajectory_epochs=4,
            trajectory_stride=2,
            trajectory_variant="both",
            trajectory_window=6,
            trajectory_window_stride=3,
            trajectory_patience=2,
            execution_mode="single",
            device_arg=device_arg,
            simulation_backend=sim_backend,
            n=4,
            n_traj=4,
            t_end=0.2,
            dt=0.02,
        )
        html_both = out_both.read_text(encoding="utf-8")
        check("both report contains v2 section", reports.ADAM_TRAJECTORY_V2_NAME in html_both)
        check("both report contains legacy section", reports.ADAM_TRAJECTORY_LEGACY_NAME in html_both)


def main(device_arg="cpu"):
    device = sindy_torch.get_device(device_arg)
    print(f"Device: {device}")
    test_v2_training(device)
    test_report_smoke(device)

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("All spring-grid trajectory-v2 verifications PASSED.")
    else:
        print("Some spring-grid trajectory-v2 verifications FAILED; see above.")
    print(f"{'=' * 50}")
    return FAIL_COUNT == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sindy_torch.add_device_arg(parser)
    args = parser.parse_args()
    success = main(args.device)
    sys.exit(0 if success else 1)
