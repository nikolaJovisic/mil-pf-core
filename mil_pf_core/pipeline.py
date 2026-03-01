from dataclasses import dataclass

from mil_pf_core.embedding.interface import EmbeddingInterface
from mil_pf_core.head.interface import HeadInterface
from mil_pf_core.preprocessing.interface import PreprocessingInterface
from mil_pf_core.types.predictions import Predictions
from mil_pf_core.types.structured_embeddings import StructuredEmbeddings
from mil_pf_core.types.view_set_list import ViewSetList


@dataclass
class Pipeline:
    preprocessing: PreprocessingInterface
    embedding: EmbeddingInterface
    head: HeadInterface

    def run(self, viewset_list: ViewSetList) -> Predictions:
        preprocessed_views = self.preprocessing.preprocess(viewset_list)
        embeddings = self.embedding.embed(preprocessed_views.images)
        structured_embeddings = StructuredEmbeddings(
            embeddings=embeddings,
            instance_types=preprocessed_views.instance_types,
            groups=preprocessed_views.groups,
        )
        return self.head.predict(structured_embeddings)

    def __call__(self, viewset_list: ViewSetList) -> Predictions:
        return self.run(viewset_list)
