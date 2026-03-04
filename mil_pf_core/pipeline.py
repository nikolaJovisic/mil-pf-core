from dataclasses import dataclass

from mil_pf_core.embedder.interface import EmbedderInterface
from mil_pf_core.head.interface import HeadInterface
from mil_pf_core.preprocessing.interface import PreprocessingInterface
from mil_pf_core.types.predictions import Predictions
from mil_pf_core.types.structured_embeddings import StructuredEmbeddings
from mil_pf_core.types.view_set_list import ViewSetList


@dataclass
class Pipeline:
    preprocessing: PreprocessingInterface
    embedder: EmbedderInterface
    head: HeadInterface

    def __post_init__(self):
        preprocessing_shape = self.preprocessing.output_shape
        embedder_shape = self.embedder.input_shape
        if (
            preprocessing_shape is not None
            and embedder_shape is not None
            and tuple(preprocessing_shape) != tuple(embedder_shape)
        ):
            raise ValueError(
                "Preprocessing output shape and embedder input shape must match: "
                f"{preprocessing_shape} != {embedder_shape}."
            )

    def run(self, viewset_list: ViewSetList) -> Predictions:
        preprocessed_views = self.preprocessing.preprocess(viewset_list)
        embeddings = self.embedder.embed(preprocessed_views.images)
        structured_embeddings = StructuredEmbeddings(
            embeddings=embeddings,
            instance_types=preprocessed_views.instance_types,
            groups=preprocessed_views.groups,
        )
        return self.head.predict(structured_embeddings)

    def __call__(self, viewset_list: ViewSetList) -> Predictions:
        return self.run(viewset_list)
