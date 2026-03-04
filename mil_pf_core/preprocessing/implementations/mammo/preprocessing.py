from typing import List

import numpy as np
import torch

from mil_pf_core.preprocessing.implementations.mammo.config import MammoPreprocessingConfig
from mil_pf_core.preprocessing.interface import PreprocessingInterface
from mil_pf_core.preprocessing.utils import center_crop_or_pad
from mil_pf_core.preprocessing.utils import preprocess_single
from mil_pf_core.preprocessing.utils import resize_img
from mil_pf_core.preprocessing.utils import tile_single
from mil_pf_core.types.images import Images
from mil_pf_core.types.preprocessed_views import PreprocessedViews
from mil_pf_core.types.view_set_list import ViewSetList


class MammoPreprocessing(PreprocessingInterface):
    def __init__(self, config: MammoPreprocessingConfig):
        self.config = config

    @property
    def output_shape(self):
        return self.config.output_shape

    def preprocess(self, viewset_list: ViewSetList) -> PreprocessedViews:
        images: List[np.ndarray] = []
        instance_types: List[int] = []
        groups: List[int] = []

        for group_idx, viewset in enumerate(viewset_list.viewsets):
            for raw_image in viewset.images:
                preprocessed = preprocess_single(
                    image=raw_image,
                    aspect_ratio=self.config.aspect_ratio,
                    shape=self.config.shape,
                    breast_mask_dilation_factor=self.config.breast_mask_dilation_factor,
                )

                full_view = resize_img(preprocessed, self.config.output_shape).astype(
                    np.float32, copy=False
                )
                images.append(full_view)
                instance_types.append(0)
                groups.append(group_idx)

                tiles = tile_single(
                    image=preprocessed,
                    tile_size=self.config.output_shape,
                    overlap=self.config.tile_overlap,
                    threshold=self.config.tile_threshold,
                    tile_increase_tolerance=self.config.tile_increase_tolerance,
                )
                for tile in tiles:
                    images.append(
                        center_crop_or_pad(tile, self.config.output_shape).astype(
                            np.float32, copy=False
                        )
                    )
                    instance_types.append(1)
                    groups.append(group_idx)

        if images:
            batch = np.stack(images, axis=0).astype(np.float32, copy=False)
        else:
            batch = np.empty(
                (0, self.config.output_shape[0], self.config.output_shape[1]),
                dtype=np.float32,
            )

        return PreprocessedViews(
            images=Images(images=torch.from_numpy(batch)),
            instance_types=torch.tensor(instance_types, dtype=torch.long),
            groups=torch.tensor(groups, dtype=torch.long),
        )
