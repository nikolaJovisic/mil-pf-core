import numpy as np

from mil_pf_core.head.interface import HeadInterface
from mil_pf_core.types.predictions import Predictions
from mil_pf_core.types.structured_embeddings import StructuredEmbeddings


class DummyHead(HeadInterface):
    def __init__(self, heatmap_height: int = 8, heatmap_width: int = 8):
        if heatmap_height <= 0 or heatmap_width <= 0:
            raise ValueError("heatmap_height and heatmap_width must be > 0.")
        self.heatmap_height = heatmap_height
        self.heatmap_width = heatmap_width

    def predict(self, structured_embeddings: StructuredEmbeddings) -> Predictions:
        batch_size = structured_embeddings.embeddings.embeddings.shape[0]
        suspicious = (np.random.rand(batch_size) > 0.5).astype(np.bool_)
        heatmap = np.random.rand(
            batch_size, self.heatmap_height, self.heatmap_width
        ).astype(np.float32)
        return Predictions(suspicious=suspicious, heatmap=heatmap)
