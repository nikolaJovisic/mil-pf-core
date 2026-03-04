import numpy as np

from mil_pf_core.preprocessing import MammoPreprocessing
from mil_pf_core.preprocessing import MammoPreprocessingConfig
from mil_pf_core.types import ViewSet
from mil_pf_core.types import ViewSetList


def test_mammo_preprocessing_outputs_views_and_tiles():
    config = MammoPreprocessingConfig(
        shape=(64, 64),
        output_shape=(16, 16),
        tile_threshold=0.0,
    )
    preprocessing = MammoPreprocessing(config=config)

    viewset_list = ViewSetList(
        viewsets=[
            ViewSet(images=[np.full((20, 12), 1000, dtype=np.uint16)]),
            ViewSet(
                images=[
                    np.full((18, 14), 2000, dtype=np.uint16),
                    np.full((22, 10), 1500, dtype=np.uint16),
                ]
            ),
        ]
    )

    out = preprocessing.preprocess(viewset_list)

    expected_views = 3
    assert out.images.images.ndim == 3
    assert out.images.images.shape[1:] == (16, 16)
    assert out.instance_types.shape[0] == out.images.images.shape[0]
    assert out.groups.shape[0] == out.images.images.shape[0]
    assert int((out.instance_types == 0).sum().item()) == expected_views
    assert out.images.images.shape[0] >= expected_views
    assert set(out.instance_types.tolist()).issubset({0, 1})
    assert set(out.groups.tolist()).issubset({0, 1})
