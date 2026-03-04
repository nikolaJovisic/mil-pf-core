from abc import ABC, abstractmethod
from typing import Tuple

from mil_pf_core.types.preprocessed_views import PreprocessedViews
from mil_pf_core.types.view_set_list import ViewSetList


class PreprocessingInterface(ABC):
    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, int]:
        """Produced (height, width) for preprocessed image tensors."""

    @abstractmethod
    def preprocess(self, viewset_list: ViewSetList) -> PreprocessedViews:
        """Maps a ViewSetList into a PreprocessedViews."""
