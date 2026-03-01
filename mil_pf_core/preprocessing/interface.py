from abc import ABC, abstractmethod

from mil_pf_core.types.preprocessed_views import PreprocessedViews
from mil_pf_core.types.view_set_list import ViewSetList


class PreprocessingInterface(ABC):
    @abstractmethod
    def preprocess(self, viewset_list: ViewSetList) -> PreprocessedViews:
        """Maps a ViewSetList into a PreprocessedViews."""
