import torch

from mil_pf_core.preprocessing.implementations.dummy.config import DummyPreprocessingConfig
from mil_pf_core.preprocessing.interface import PreprocessingInterface
from mil_pf_core.types.images import Images
from mil_pf_core.types.preprocessed_views import PreprocessedViews
from mil_pf_core.types.view_set_list import ViewSetList


class DummyPreprocessing(PreprocessingInterface):
    def __init__(self, config: DummyPreprocessingConfig):
        self.config = config

    @property
    def output_shape(self):
        return self.config.output_shape

    def preprocess(self, viewset_list: ViewSetList) -> PreprocessedViews:
        groups = []
        instance_types = []

        for group_idx, viewset in enumerate(viewset_list.viewsets):
            for _ in viewset.images:
                groups.extend([group_idx] * (1 + self.config.tiles_per_image))
                instance_types.extend([0] + [1] * self.config.tiles_per_image)

        batch_size = len(groups)
        images = torch.rand(
            batch_size, self.config.output_shape[0], self.config.output_shape[1]
        )

        return PreprocessedViews(
            images=Images(images=images),
            instance_types=torch.tensor(instance_types, dtype=torch.long),
            groups=torch.tensor(groups, dtype=torch.long),
        )
