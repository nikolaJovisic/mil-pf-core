# mil-pf-core

Core pipeline package for MIL-PF preprocessing, embedding, and head inference.

MIL-PF is a mammography classification approach relying on foundational vision encoders and custom Mulitple Instance Learning (MIL) attention-based aggregation head. It is acheving state-of-the-art classification results, while reducing training time from hours to minutes compared to the baselines.

## Project context

- This repository evolved from [`mil-pf-research`](https://github.com/nikolaJovisic/mil-pf-research).
- This repository is consumed by [`mil-pf-serve`](https://github.com/nikolaJovisic/mil-pf-serve).
- Heatmap behavior is still in migration from the research implementation to the production implementation.

## Installation

Use `uv` in this repository:

```bash
uv sync
```

## Running tests

Quick test set:

```bash
uv run pytest -r tests/test_dummy_pipeline.py tests/test_pipeline_shape_compatibility.py tests/test_mammo_preprocessing.py
```

Verbose real-components test:

```bash
uv run pytest -s tests/test_real_pipeline_verbose.py
```

## Tested real pipeline configuration

The currently tested real integration path is:

- Preprocessing: `MammoPreprocessing`
- Embedder: `DinoV2Embedder` using `facebookresearch/dinov2` / `dinov2_vitg14`
- Head: `MILPFAttnHead` with external checkpoint

## Minimal usage sketch

```python
from mil_pf_core.pipeline import Pipeline
from mil_pf_core.preprocessing import MammoPreprocessing, MammoPreprocessingConfig
from mil_pf_core.embedder import DinoV2Embedder, DinoV2EmbedderConfig
from mil_pf_core.head import MILPFAttnHead, MILPFAttnHeadConfig, MILPFAttnModelConfig

pipeline = Pipeline(
    preprocessing=MammoPreprocessing(MammoPreprocessingConfig(output_shape=(518, 518))),
    embedder=DinoV2Embedder(DinoV2EmbedderConfig(input_shape=(518, 518))),
    head=MILPFAttnHead(MILPFAttnHeadConfig(model=MILPFAttnModelConfig(), head_path="/path/to/weights.pth")),
)
```
