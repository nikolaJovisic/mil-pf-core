import pytest

from mil_pf_core.embedder import DummyEmbedder
from mil_pf_core.embedder import DummyEmbedderConfig
from mil_pf_core.head import DummyHead
from mil_pf_core.head import DummyHeadConfig
from mil_pf_core.pipeline import Pipeline
from mil_pf_core.preprocessing import DummyPreprocessing
from mil_pf_core.preprocessing import DummyPreprocessingConfig


def test_pipeline_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        Pipeline(
            preprocessing=DummyPreprocessing(
                config=DummyPreprocessingConfig(output_shape=(16, 16))
            ),
            embedder=DummyEmbedder(
                config=DummyEmbedderConfig(embedding_dim=8, input_shape=(32, 32))
            ),
            head=DummyHead(config=DummyHeadConfig(heatmap_shape=(8, 8))),
        )


def test_pipeline_allows_matching_shapes():
    pipeline = Pipeline(
        preprocessing=DummyPreprocessing(
            config=DummyPreprocessingConfig(output_shape=(16, 16))
        ),
        embedder=DummyEmbedder(
            config=DummyEmbedderConfig(embedding_dim=8, input_shape=(16, 16))
        ),
        head=DummyHead(config=DummyHeadConfig(heatmap_shape=(8, 8))),
    )
    assert pipeline is not None
