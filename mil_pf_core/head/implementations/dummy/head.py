import numpy as np

from mil_pf_core.head.implementations.dummy.config import DummyHeadConfig
from mil_pf_core.head.interface import HeadInterface
from mil_pf_core.types.predictions import Predictions
from mil_pf_core.types.structured_embeddings import StructuredEmbeddings


class DummyHead(HeadInterface):
    def __init__(self, config: DummyHeadConfig):
        self.config = config

    def predict(self, structured_embeddings: StructuredEmbeddings) -> Predictions:
        batch_size = structured_embeddings.embeddings.embeddings.shape[0]
        suspicious = (
            np.random.rand(batch_size) > self.config.suspicious_threshold
        ).astype(np.bool_)
        heatmap = np.random.rand(
            batch_size, self.config.heatmap_shape[0], self.config.heatmap_shape[1]
        ).astype(np.float32)
        return Predictions(suspicious=suspicious, heatmap=heatmap)
