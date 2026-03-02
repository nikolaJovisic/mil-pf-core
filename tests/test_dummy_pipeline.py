import numpy as np

from mil_pf_core.embedding import DummyEmbedding
from mil_pf_core.head import DummyHead
from mil_pf_core.pipeline import Pipeline
from mil_pf_core.preprocessing import DummyPreprocessing
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
            output_height=16,
            output_width=16,
            tiles_per_image=tiles_per_image,
        ),
        embedding=DummyEmbedding(embedding_dim=16),
        head=DummyHead(heatmap_height=8, heatmap_width=8),
    )

    output = pipeline(viewset_list)

    expected_batch = 3 * (1 + tiles_per_image)

    assert isinstance(output, Predictions)
    assert output.suspicious.dtype == np.bool_
    assert output.suspicious.ndim == 1
    assert output.suspicious.shape[0] == expected_batch
    assert output.heatmap.shape == (expected_batch, 8, 8)
