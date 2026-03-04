import numpy as np

from mil_pf_core.embedder import DummyEmbedder
from mil_pf_core.embedder import DummyEmbedderConfig
from mil_pf_core.head import DummyHead
from mil_pf_core.head import DummyHeadConfig
from mil_pf_core.pipeline import Pipeline
from mil_pf_core.preprocessing import DummyPreprocessing
from mil_pf_core.preprocessing import DummyPreprocessingConfig
from mil_pf_core.types import Predictions
from mil_pf_core.types import ViewSet
from mil_pf_core.types import ViewSetList


def test_dummy_pipeline_smoke():
    tiles_per_image = 6
    viewset_list = ViewSetList(
        viewsets=[
            ViewSet(images=[np.ones((10, 10), dtype=np.uint16), np.ones((10, 10), dtype=np.uint16)]),
            ViewSet(images=[np.ones((12, 8), dtype=np.uint16)]),
        ]
    )

    pipeline = Pipeline(
        preprocessing=DummyPreprocessing(
            config=DummyPreprocessingConfig(
                output_shape=(16, 16),
                tiles_per_image=tiles_per_image,
            )
        ),
        embedder=DummyEmbedder(
            config=DummyEmbedderConfig(embedding_dim=16, input_shape=(16, 16))
        ),
        head=DummyHead(config=DummyHeadConfig(heatmap_shape=(8, 8))),
    )

    output = pipeline(viewset_list)

    expected_batch = 3 * (1 + tiles_per_image)

    assert isinstance(output, Predictions)
    assert output.suspicious.dtype == np.bool_
    assert output.suspicious.ndim == 1
    assert output.suspicious.shape[0] == expected_batch
    assert output.heatmap.shape == (expected_batch, 8, 8)
