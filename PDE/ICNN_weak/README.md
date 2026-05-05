# ICNN_weak

Weak-form (WSINDy-style) ICNN training pipeline.

Uses K=50 random raised-cosine test functions and a separate
`rxn_factor`-weighted reaction force loss term.

## Files
- `nn_model.py`  — ICNN architecture (same as ICNN/)
- `pde_loss.py`  — weak-form loss with random test functions
- `train.py`     — L-BFGS trainer with Adam warm-start
- `run.py`       — full pipeline with 2-phase training
- `smoothing.py` — Gaussian kernel smoother (shared)

## Known issues
- `rxn_factor` needs hand-tuning; wrong value causes plateau or NaN
- L-BFGS crashes with dropout > 0
- Weak-form and reaction loss are in different units — scale conflict

See `../ICNN/` for the cleaner nn-EUCLID nodal-force implementation.
