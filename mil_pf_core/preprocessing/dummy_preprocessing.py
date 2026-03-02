import torch

from mil_pf_core.preprocessing.interface import PreprocessingInterface
from mil_pf_core.types.images import Images
from mil_pf_core.types.preprocessed_views import PreprocessedViews
from mil_pf_core.types.view_set_list import ViewSetList


class DummyPreprocessing(PreprocessingInterface):
    def __init__(
        self,
        output_height: int = 32,
        output_width: int = 32,
        tiles_per_image: int = 6,
    ):
        if output_height <= 0 or output_width <= 0:
            raise ValueError("output_height and output_width must be > 0.")

        self.output_height = output_height
        self.output_width = output_width
        self.tiles_per_image = tiles_per_image

    def preprocess(self, viewset_list: ViewSetList) -> PreprocessedViews:
        groups = []
        instance_types = []

        for group_idx, viewset in enumerate(viewset_list.viewsets):
            for _ in viewset.images:
                groups.extend([group_idx] * (1 + self.tiles_per_image))
                instance_types.extend([0] + [1] * self.tiles_per_image)

        batch_size = len(groups)
        images = torch.rand(batch_size, self.output_height, self.output_width)

        return PreprocessedViews(
            images=Images(images=images),
            instance_types=torch.tensor(instance_types, dtype=torch.long),
            groups=torch.tensor(groups, dtype=torch.long),
        )
