# Repository Instructions

- Always use `einops` for tensor reshape/repeat/rearrange operations instead of manual `view`, `reshape`, `permute`, `unsqueeze`, or `repeat` chains where applicable.

## Repository context

- This repository evolved from `mil-pf-research`.
- This repository is used by `mil-pf-serve`.
- `MILPFAttnHead` outputs case-level predictions (group-level), not per-instance predictions.
- Heatmap behavior is currently being migrated from research behavior to production behavior.

## Testing notes

- The real verbose integration test is `tests/test_real_pipeline_verbose.py`.
- The tested DINOv2 setup in this repository is:
  - repo: `facebookresearch/dinov2`
  - model: `dinov2_vitg14`
- When changing embedding or head configs, keep embedding dimension compatible with checkpoint head input dimension.
