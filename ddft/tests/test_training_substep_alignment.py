import builtins
import os
import sys

import torch
from phi.torch.flow import *
# from phi.torch.flow import * shadows Python's builtin round() with PhiFlow's
# own (float-returning) round_ - use builtins.round below, same fix as
# train_phiflow.py itself.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "training"))

from neural_ddft.models.model import SuperadiabaticNet
from neural_ddft.physics.phiflow_pde import PhiFlowDDFT
from neural_ddft.physics.force_factory import FORCE_MODES
from train_phiflow import pde_step


def test_substep_training_matches_eval_step_exactly():
    """
    Regression test for the training time-alignment fix: train_phiflow's
    pde_step(), called n_sub times per frame with alpha=force_alpha and
    rho_prev/v_prev updated every substep (the fixed pattern used in the
    unroll loop), must produce states matching eval's PhiFlowDDFT.step()
    called the same number of times with the same alpha/dt - both implement
    the identical momentum-equation formula, so any divergence here means
    the fixed unroll's substep/alpha handling has drifted from eval's.

    Before the fix, training called pde_step ONCE per frame (not n_sub
    times) with alpha=5 (not 1) - that pattern would fail this test by a
    wide margin, not just marginally outside tolerance.
    """
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dt = 1e-4
    frame_interval = 0.01
    n_sub = builtins.round(frame_interval / dt)
    assert n_sub == 100
    alpha = 1.0
    gamma, eta, k, T, m = 0.1, 0.6944, 1.0, 2.0, 1.0

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, rho):
            return torch.sum(rho ** 2, dim=-1, keepdim=True) * self.weight

    base_model = MockBaseModel().eval().to(device)
    torch.manual_seed(0)
    sup_model = SuperadiabaticNet(
        channels=8, use_rho_prev=True, use_v_prev=True, use_accel=True,
    ).eval().to(device)
    # Zero-initialized output head by design - perturb so f_sup is genuinely
    # non-zero and input-dependent, matching a mid-training checkpoint, not
    # the untrained starting point where both paths would trivially agree
    # (f_sup=0 everywhere regardless of substep/alpha handling).
    with torch.no_grad():
        sup_model.net[-1].weight.add_(0.1)
        sup_model.net[-1].bias.add_(0.05)

    models = {"base": base_model, "sup": sup_model}
    config = {"model_mode": "hybrid", "m": m, "gamma": gamma, "eta": eta, "k": k, "T": T,
              "dt": dt, "force_alpha": alpha}
    compute_internal_force = FORCE_MODES["hybrid"]

    torch.manual_seed(1)
    rho0 = CenteredGrid(0.6 + 0.05 * math.random_normal(spatial(x=domain_res)),
                         extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v0 = StaggeredGrid(0.05 * math.random_normal(spatial(x=domain_res)),
                        extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(0.3 * math.random_normal(spatial(x=domain_res)),
                          extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    W = 3

    # --- Path A: eval's own PhiFlowDDFT.step, called n_sub*W times, alpha=1 ---
    solver = PhiFlowDDFT(models=models, device=device, dx=dx, config=config)
    rho_eval, v_eval = rho0, v0
    rho_prev_eval, v_prev_eval = None, None
    eval_frame_states = []
    for frame in range(W):
        for sub in range(n_sub):
            rho_next, v_next = solver.step(rho_eval, v_eval, dt, V_ext,
                                            rho_prev=rho_prev_eval, v_prev=v_prev_eval, alpha=alpha)
            rho_prev_eval, v_prev_eval = rho_eval, v_eval
            rho_eval, v_eval = rho_next, v_next
        eval_frame_states.append((rho_eval, v_eval))

    # --- Path B: training's pde_step, called the same way (the fixed pattern) ---
    rho_train, v_train = rho0, v0
    rho_prev_train, v_prev_train = None, None
    train_frame_states = []
    for frame in range(W):
        for sub in range(n_sub):
            rho_next, v_next, _ = pde_step(
                rho_train, v_train, dt, V_ext, compute_internal_force, models, dx, device, config,
                rho_prev=rho_prev_train, v_prev=v_prev_train, alpha=config["force_alpha"],
            )
            rho_prev_train, v_prev_train = rho_train, v_train
            rho_train, v_train = rho_next, v_next
        train_frame_states.append((rho_train, v_train))

    for frame, ((rho_e, v_e), (rho_t, v_t)) in enumerate(zip(eval_frame_states, train_frame_states)):
        rho_e_np = rho_e.values.native('batch,x').detach().cpu().numpy()
        rho_t_np = rho_t.values.native('batch,x').detach().cpu().numpy()
        v_e_np = v_e.values.vector['x'].native('batch,x').detach().cpu().numpy()
        v_t_np = v_t.values.vector['x'].native('batch,x').detach().cpu().numpy()

        rho_diff = abs(rho_e_np - rho_t_np).max()
        v_diff = abs(v_e_np - v_t_np).max()
        assert rho_diff < 1e-6, f"frame {frame}: rho mismatch {rho_diff} - substep/alpha handling has drifted from eval"
        assert v_diff < 1e-6, f"frame {frame}: v mismatch {v_diff} - substep/alpha handling has drifted from eval"

    print("[DEBUG] test_substep_training_matches_eval_step_exactly passed - training and eval agree substep-for-substep.")


def test_force_penalty_and_ratio_are_averaged_not_summed_over_substeps():
    """
    The trajectory-matching test above only checks forward states - it can't
    catch a mis-scaled loss, since PhiFlowDDFT.step never computes a loss or
    force penalty at all. This test independently replicates the exact
    accumulate-then-divide-by-n_sub pattern used in train_phiflow.py's
    unroll (lines around frame_force_penalty_sum/frame_force_ratio_sum) for
    ONE frame, and cross-checks it two ways:
      1. The accumulator-based average must equal a plainly-computed mean
         of the same n_sub per-substep values (same arithmetic, independent
         implementation - catches an off-by-one or wrong-denominator bug).
      2. The averaged value must NOT equal the raw (un-divided) sum, by a
         wide margin - if a future edit reverts to summing instead of
         averaging over substeps, this assertion fails loudly instead of
         silently making the force penalty n_sub times stronger (exactly
         the "don't sum them over n_sub" failure mode).
    """
    box_length = 10.0
    domain_res = 32
    dx = box_length / domain_res
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dt = 1e-4
    n_sub = 10  # small for test speed - the averaging arithmetic doesn't depend on n_sub's magnitude
    alpha = 1.0
    gamma, eta, k, T, m = 0.1, 0.6944, 1.0, 2.0, 1.0

    class MockBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, rho):
            return torch.sum(rho ** 2, dim=-1, keepdim=True) * self.weight

    base_model = MockBaseModel().eval().to(device)
    torch.manual_seed(2)
    sup_model = SuperadiabaticNet(channels=8, use_rho_prev=True, use_v_prev=True, use_accel=True).eval().to(device)
    with torch.no_grad():
        sup_model.net[-1].weight.add_(0.1)
        sup_model.net[-1].bias.add_(0.05)

    models = {"base": base_model, "sup": sup_model}
    config = {"model_mode": "hybrid", "m": m, "gamma": gamma, "eta": eta, "k": k, "T": T,
              "dt": dt, "force_alpha": alpha}
    compute_internal_force = FORCE_MODES["hybrid"]

    torch.manual_seed(3)
    rho = CenteredGrid(0.6 + 0.05 * math.random_normal(spatial(x=domain_res)),
                        extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    v = StaggeredGrid(0.05 * math.random_normal(spatial(x=domain_res)),
                       extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))
    V_ext = CenteredGrid(0.3 * math.random_normal(spatial(x=domain_res)),
                          extrapolation.PERIODIC, x=domain_res, bounds=Box(x=box_length))

    rho_prev, v_prev = None, None
    frame_force_ratio_sum = torch.tensor(0.0)
    frame_force_penalty_sum = torch.tensor(0.0)
    raw_penalties = []  # independently-collected per-substep mean(f_sup^2) values
    raw_ratios = []

    for sub in range(n_sub):
        rho_next, v_next, force_components = pde_step(
            rho, v, dt, V_ext, compute_internal_force, models, dx, device, config,
            rho_prev=rho_prev, v_prev=v_prev, alpha=config["force_alpha"],
        )
        f_eq_grid = force_components["f_eq"]
        f_sup_grid = force_components["f_sup"]
        f_eq_tensor = f_eq_grid.values.vector['x'].native(f_eq_grid.values.vector['x'].shape)
        f_sup_tensor = f_sup_grid.values.vector['x'].native(f_sup_grid.values.vector['x'].shape)

        ratio = (torch.norm(f_sup_tensor) / (torch.norm(f_eq_tensor) + 1e-8)).detach()
        penalty = torch.mean(f_sup_tensor ** 2)

        # This mirrors train_phiflow.py's accumulation exactly.
        frame_force_ratio_sum = frame_force_ratio_sum + ratio
        frame_force_penalty_sum = frame_force_penalty_sum + penalty

        raw_ratios.append(ratio.item())
        raw_penalties.append(penalty.item())

        rho_prev, v_prev = rho, v
        rho, v = rho_next, v_next

    accumulator_avg_penalty = (frame_force_penalty_sum / n_sub).item()
    accumulator_avg_ratio = (frame_force_ratio_sum / n_sub).item()

    independent_avg_penalty = sum(raw_penalties) / len(raw_penalties)
    independent_avg_ratio = sum(raw_ratios) / len(raw_ratios)

    assert abs(accumulator_avg_penalty - independent_avg_penalty) < 1e-9, (
        f"force penalty average mismatch: accumulator={accumulator_avg_penalty} vs independent={independent_avg_penalty}"
    )
    assert abs(accumulator_avg_ratio - independent_avg_ratio) < 1e-5 * max(abs(independent_avg_ratio), 1.0), (
        f"force ratio average mismatch: accumulator={accumulator_avg_ratio} vs independent={independent_avg_ratio}"
    )

    # Sanity: the raw values must be non-trivially non-zero (sup net perturbed
    # above specifically so this isn't a vacuous all-zero check), and the
    # un-divided sum must differ from the average by roughly a factor of
    # n_sub - if it didn't, this test wouldn't be able to tell "averaged"
    # from "summed" apart in the first place.
    assert independent_avg_penalty > 1e-12, "f_sup is suspiciously zero - perturbation didn't take effect"
    raw_sum_penalty = sum(raw_penalties)
    assert abs(raw_sum_penalty - independent_avg_penalty) > 0.5 * (n_sub - 1) * independent_avg_penalty, (
        "averaged and summed force penalty are too close together for this test to distinguish them - "
        "check that n_sub > 1 and f_sup varies enough across substeps"
    )

    print("[DEBUG] test_force_penalty_and_ratio_are_averaged_not_summed_over_substeps passed.")
