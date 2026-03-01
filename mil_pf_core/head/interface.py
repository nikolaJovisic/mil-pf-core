from abc import ABC, abstractmethod

from mil_pf_core.types.predictions import Predictions
from mil_pf_core.types.structured_embeddings import StructuredEmbeddings


class HeadInterface(ABC):
    @abstractmethod
    def predict(self, structured_embeddings: StructuredEmbeddings) -> Predictions:
        """Maps StructuredEmbeddings into Predictions."""
