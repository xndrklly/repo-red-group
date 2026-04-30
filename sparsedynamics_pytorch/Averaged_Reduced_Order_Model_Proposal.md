# Averaged Reduced-Order Model Proposal

## Goal

Build a coarse 5x5 spring-grid model from the current 10x10 model by averaging each 2x2 block of fine-grid motion into one coarse-grid state, then reuse the existing regression and trajectory-loss machinery on the coarse data.

This proposal is written against the current workflow in:

- `examples/spring_grid_K_regression.py`
- `examples/generate_spring_grid_three_model_reports.py`
- `sindy_torch/trajectory.py`


## 1. Current fine-grid workflow

The current spring-grid pipeline does three important things already:

1. It generates fine-grid trajectories with displacement, velocity, acceleration, forcing, and the corrected regression target:

   `A_target = A + V C^T - F`

   so that, in the zero-damping / zero-forcing case used by the default examples,

   `A_target = A = -U K_free^T`.

2. It builds a regression problem of the form

   `target ~= theta @ Xi`

   with:

   - `theta = U`
   - `target = A_target`
   - `K_pred = -Xi^T`

3. It optionally refines the model with the trajectory-v2 loss by fitting a symmetric local stiffness matrix on windowed `[u, v]` trajectories using normalized displacement/velocity MSE.

That structure is already the right one for a reduced-order model. The only missing piece is a coarse representation of the state and a matching coarse operator.


## 2. Coarse state definition

### 2.1 Coarse grid

For a 10x10 fine grid, define a 5x5 coarse grid by grouping fine nodes into 2x2 blocks:

- coarse cell `(I, J)` contains fine rows `2I, 2I+1` and fine columns `2J, 2J+1`
- number of coarse displacement states: `N_rom = 25`
- number of coarse first-order states `[u_rom, v_rom]`: `2 * N_rom = 50`

### 2.2 Recommended averaging rule

The current code works on free DOFs after the clamped row is removed. Because of that, the cleanest reduced model is:

- average over the active fine DOFs inside each 2x2 block
- normalize by the number of active DOFs in that block

So for coarse cell `c`,

`u_rom,c(t) = (1 / m_c) * sum_{i in block c} u_i(t)`

`v_rom,c(t) = (1 / m_c) * sum_{i in block c} v_i(t)`

where `m_c` is the number of free fine DOFs in that block.

This gives:

- interior blocks: `m_c = 4`
- blocks touching the clamped row: `m_c = 2`

This is the recommended definition because it preserves the "average motion in a block" idea while remaining consistent with the existing free-DOF formulation.

### 2.3 Restriction matrix

Define a coarse restriction matrix

`R in R^(25 x 90)`

for the current 10x10, one-pinned-row case. Each row of `R` corresponds to one coarse block:

- `R[c, i] = 1 / m_c` if fine free DOF `i` belongs to block `c`
- `R[c, i] = 0` otherwise

Then the coarse data are:

`U_rom = U_free @ R^T`

`V_rom = V_free @ R^T`

`A_rom = A_free @ R^T`

`F_rom = F_free @ R^T`

`A_target_rom = A_target @ R^T`

The same restriction should be applied trajectory-by-trajectory to produce coarse trajectory dictionaries.


## 3. Reduced model

There are two compatible reduced models to use together:

### 3.1 Projected coarse model from the current fine model

If `K_current` is the current fine-grid stiffness matrix, define a block-constant prolongation

`P in R^(90 x 25)`

with:

- `P[i, c] = 1` if fine free DOF `i` belongs to coarse block `c`
- `P[i, c] = 0` otherwise

Using the active-node-normalized restriction above gives `R P = I`, so a projected coarse operator is

`K_rom_proj = R K_current P`

and, if damping is used later,

`C_rom_proj = R C_current P`.

This is the direct "average of the current model" piece.

It gives a physically interpretable coarse operator and a good warm start for optimization.

### 3.2 Learned coarse model from coarse data

The reduced regression model keeps exactly the same form as the current one:

`A_target_rom ~= U_rom @ Xi_rom`

with

`K_rom = -Xi_rom^T`.

This is important: the reduced model is not just the averaged `K`; it is the best coarse operator that explains the averaged trajectories under the current loss functions.


## 4. Locality structure on the coarse grid

The current fine model uses an 8-neighbor locality mask plus self interaction. The reduced model should do the same on the 5x5 grid:

- self
- left / right
- up / down
- both diagonals

So the coarse regression mask should be:

`locality_rom = build_locality_mask(5, include_anti_diagonal=True)`

This keeps the same structural prior as the current regression problem, just at the coarse scale.


## 5. Reduced data to feed into the current methods

The reduced dataset should mirror the current `generate_dataset(...)` output, but with coarse arrays added.

Recommended added fields:

- `R_avg`: coarse restriction matrix
- `n_rom`: coarse side length, here `5`
- `N_rom`: total coarse DOFs, here `25`
- `U_rom`: pooled coarse displacements
- `V_rom`: pooled coarse velocities
- `A_rom`: pooled coarse accelerations
- `F_rom`: pooled coarse forcing
- `A_target_rom`: pooled coarse corrected acceleration target
- `locality_rom`: 5x5-grid locality mask

Each trajectory entry should also carry:

- `U_rom`
- `V_rom`
- `A_rom`
- `F_rom`
- `A_target_rom`
- `u0_rom`
- `v0_rom`

This keeps the current regression and trajectory code paths nearly unchanged.


## 6. How it plugs into the current regression methods

### 6.1 Residual regression

The current residual methods can be reused with:

- `theta_train = U_rom[train_idx]`
- `target_train = A_target_rom[train_idx]`
- `theta_test = U_rom[test_idx]`
- `target_test = A_target_rom[test_idx]`
- `xi_mask = locality_rom`

Then the same methods apply directly:

- STLS
- Adam derivative matching
- Adam + L1
- ISTA / proximal updates

The only dimensional change is:

- fine model: `Xi in R^(90 x 90)`
- reduced model: `Xi_rom in R^(25 x 25)`

### 6.2 Warm start

There are two reasonable warm starts for the coarse regression:

1. `Xi_rom_init = -(K_rom_proj)^T`
2. coarse residual fit from zero initialization

The better practical path is:

1. project the current fine model to get `K_rom_proj`
2. convert it to `Xi_rom_init`
3. run a short coarse residual regression
4. use that output to warm-start trajectory optimization


## 7. How it plugs into the current trajectory losses

The trajectory-v2 workflow can also be reused almost unchanged.

### 7.1 Coarse trajectory state

For each reduced trajectory window, define

`x_rom(t) = [u_rom(t), v_rom(t)] in R^50`.

These windows can be built with the same helpers already used now:

- `stack_state_trajectory(...)`
- `make_overlapping_trajectory_windows(...)`
- `compute_state_scales(...)`
- `normalized_state_mse(...)`

### 7.2 Coarse rollout model

The coarse second-order model is

`u_rom_tt + C_rom u_rom_t + K_rom u_rom = f_rom`

or, in the default zero-damping / zero-forcing case,

`u_rom_tt = -K_rom u_rom`.

In first-order form:

`d/dt [u_rom, v_rom] = [v_rom, -K_rom u_rom]`.

This is exactly the same model class currently used for the fine-grid trajectory training, only with 25 displacement DOFs instead of 90.

### 7.3 Coarse trajectory loss

Use the same normalized state loss on the reduced windows:

`L_traj = normalized_state_mse(x_pred_rom, x_true_rom, ...)`

with:

- `n_position_states = 25`
- `u_scale` computed from reduced training windows
- `v_scale` computed from reduced training windows
- same 50/50 displacement/velocity weighting

This is the right reduced-order analogue of the current trajectory-v2 loss.


## 8. Recommended training recipe

The reduced-order workflow should be:

1. Generate the current fine-grid dataset as usual.
2. Build `R_avg` for 2x2 block averaging.
3. Form the reduced pooled data `U_rom`, `V_rom`, `A_target_rom`.
4. Build a projected coarse warm start `K_rom_proj = R K_current P`.
5. Run coarse residual regression with the current STLS / Adam / ISTA methods.
6. Warm-start a symmetric local coarse stiffness model from the best residual coarse fit.
7. Fine-tune with the current trajectory-v2 windowed normalized loss on coarse trajectories.

If a joint objective is desired, it should be:

`L_total = lambda_res * L_residual + lambda_traj * L_traj + lambda_1 * ||Xi_rom||_1`

where:

- `L_residual = MSE(U_rom @ Xi_rom, A_target_rom)`
- `L_traj` is the current normalized windowed trajectory loss
- the L1 term is optional, exactly as in the current code


## 9. Why this is a sensible ROM for the current repository

This proposal matches the current codebase well because it:

- uses the same data products already generated now
- keeps the same regression target definition
- keeps the same locality prior
- keeps the same symmetric-stiffness trajectory refinement idea
- reduces the learned operator from 90x90 to 25x25
- reduces the first-order trajectory state from 180 to 50

So it is a true reduced-order model, not a new learning setup.


## 10. Suggested implementation changes

The smallest clean implementation would be:

1. Add a helper to build the 2x2 block restriction / prolongation matrices.
2. Add a helper that augments the existing dataset with reduced arrays.
3. Add a reduced version of `build_regression_problem(...)` that reads `U_rom`, `A_target_rom`, and `locality_rom`.
4. Add a reduced trajectory path that builds windows from `U_rom` and `V_rom`.
5. Use `K_rom_proj` as an optional warm start for the reduced trajectory model.

Concretely, the most natural insertion points are:

- dataset augmentation near `generate_dataset(...)`
- reduced regression assembly near `build_regression_problem(...)`
- reduced window building near `build_windowed_trajectory_set(...)`
- reduced trajectory optimization as a coarse analogue of `train_trajectory_v2_method(...)`


## 11. Practical note on the top coarse row

If you want the reduced model to remain maximally compatible with the current free-DOF formulation, the top coarse row should average only the active fine nodes in each support-touching 2x2 block.

If you instead want a literal 4-node average that includes the clamped zeros, that is also possible, but then the reduced mass / support treatment should be handled explicitly rather than reusing the current identity-mass assumptions unchanged.

For this repository, the active-node-normalized average is the cleaner reduced-order choice.
