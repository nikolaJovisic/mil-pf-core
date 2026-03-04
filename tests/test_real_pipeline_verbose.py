from pathlib import Path
from typing import Dict
from typing import List

import cv2
import numpy as np
import pytest
import torch

from mil_pf_core.embedder import DinoV2Embedder
from mil_pf_core.embedder import DinoV2EmbedderConfig
from mil_pf_core.head import MILPFAttnHead
from mil_pf_core.head import MILPFAttnHeadConfig
from mil_pf_core.head import MILPFAttnModelConfig
from mil_pf_core.preprocessing import MammoPreprocessing
from mil_pf_core.preprocessing import MammoPreprocessingConfig
from mil_pf_core.types import StructuredEmbeddings
from mil_pf_core.types import ViewSet
from mil_pf_core.types import ViewSetList


TEST_IMAGES_ROOT = Path(__file__).parent / "images"
HEAD_WEIGHTS_PATH = Path(
   "/lustre/data/cvrs.mammo.ivi/nj/cvpr2026/head_weights/ad2092ef.pth"
)


def _to_uint16_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint16:
        return image
    if image.dtype == np.uint8:
        return (image.astype(np.uint16) << 8)
    return image.astype(np.uint16)


def _load_viewset_list_from_test_images(root: Path) -> ViewSetList:
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(case_dirs) == 2, f"Expected exactly 2 cases under {root}, got {len(case_dirs)}."

    viewsets: List[ViewSet] = []
    print("\n========== INPUT DATA ==========")
    print(f"Image root: {root}")
    print(f"Number of cases: {len(case_dirs)}")

    for case_idx, case_dir in enumerate(case_dirs):
        image_paths = sorted(case_dir.glob("*.png"))
        assert len(image_paths) == 2, (
            f"Expected exactly 2 images in case {case_dir.name}, got {len(image_paths)}."
        )
        print(f"Case #{case_idx} ({case_dir.name}) image count: {len(image_paths)}")

        case_images: List[np.ndarray] = []
        for image_idx, image_path in enumerate(image_paths):
            loaded = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            assert loaded is not None, f"Failed to read image: {image_path}"
            image = _to_uint16_grayscale(loaded)
            print(
                "  "
                f"[case={case_idx} image={image_idx}] path={image_path.name} "
                f"shape={image.shape} dtype={image.dtype} min={int(image.min())} max={int(image.max())}"
            )
            case_images.append(image)

        viewsets.append(ViewSet(images=case_images))

    return ViewSetList(viewsets=viewsets)


def _extract_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a dict-like state_dict.")
    return state_dict


def _infer_head_model_config(checkpoint_path: Path) -> MILPFAttnModelConfig:
    state_dict = _extract_state_dict(checkpoint_path)

    global_proj_w = state_dict["global_proj_0.0.weight"]
    local_proj_w = state_dict["local_proj_0.0.weight"]
    latent = state_dict["latent"]

    input_dim = int(global_proj_w.shape[1])
    gl_hidden_dim = int(global_proj_w.shape[0] // 2)
    lc_hidden_dim = int(local_proj_w.shape[0] // 2)
    num_latents = int(latent.shape[0])

    if "linear_out.weight" in state_dict:
        mlp_out = False
        num_classes = int(state_dict["linear_out.weight"].shape[0])
    else:
        mlp_out = True
        num_classes = int(state_dict["linear_out.2.weight"].shape[0])

    return MILPFAttnModelConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        gl_hidden_dim=gl_hidden_dim,
        lc_hidden_dim=lc_hidden_dim,
        num_latents=num_latents,
        mlp_out=mlp_out,
    )


def test_real_pipeline_verbose_with_mammo_dinov2_and_milpfattn():
    if not HEAD_WEIGHTS_PATH.exists():
        pytest.skip(f"Missing head weights at: {HEAD_WEIGHTS_PATH}")

    assert TEST_IMAGES_ROOT.exists(), f"Missing test image root: {TEST_IMAGES_ROOT}"
    viewset_list = _load_viewset_list_from_test_images(TEST_IMAGES_ROOT)

    print("\n========== PREPROCESSING ==========")
    preprocessing = MammoPreprocessing(
        config=MammoPreprocessingConfig(
            shape=(1024, 1024),
            output_shape=(518, 518),
            tile_threshold=0.2,
        )
    )
    preprocessed = preprocessing.preprocess(viewset_list)
    pre_images = preprocessed.images.images
    print(f"MammoPreprocessing output images shape: {tuple(pre_images.shape)}")
    print(f"MammoPreprocessing output images dtype: {pre_images.dtype}")
    print(f"instance_types shape: {tuple(preprocessed.instance_types.shape)}")
    print(f"groups shape: {tuple(preprocessed.groups.shape)}")
    print(f"whole views count (instance_type=0): {int((preprocessed.instance_types == 0).sum().item())}")
    print(f"tiles count (instance_type=1): {int((preprocessed.instance_types == 1).sum().item())}")
    for case_idx in range(len(viewset_list.viewsets)):
        case_items = int((preprocessed.groups == case_idx).sum().item())
        print(f"group(case)={case_idx} preprocessed items: {case_items}")

    assert pre_images.ndim == 3
    assert tuple(pre_images.shape[1:]) == (518, 518)
    assert pre_images.shape[0] > 0

    print("\n========== EMBEDDING ==========")
    embedder = DinoV2Embedder(
        config=DinoV2EmbedderConfig(
            input_shape=(518, 518),
            device="cpu",
        )
    )

    embeddings = embedder.embed(preprocessed.images)
    emb = embeddings.embeddings
    print(f"DinoV2 input images shape: {tuple(pre_images.shape)}")
    print(f"DinoV2 output embeddings shape: {tuple(emb.shape)}")
    print(f"DinoV2 output embeddings dtype: {emb.dtype}")

    assert emb.ndim == 2
    assert emb.shape[0] == pre_images.shape[0]

    print("\n========== HEAD ==========")
    inferred_model_config = _infer_head_model_config(HEAD_WEIGHTS_PATH)
    print("Inferred MILPFAttn model config from checkpoint:")
    print(f"  input_dim={inferred_model_config.input_dim}")
    print(f"  num_classes={inferred_model_config.num_classes}")
    print(f"  gl_hidden_dim={inferred_model_config.gl_hidden_dim}")
    print(f"  lc_hidden_dim={inferred_model_config.lc_hidden_dim}")
    print(f"  num_latents={inferred_model_config.num_latents}")
    print(f"  mlp_out={inferred_model_config.mlp_out}")

    assert emb.shape[1] == inferred_model_config.input_dim, (
        "Embedder output dim does not match head checkpoint input_dim: "
        f"{emb.shape[1]} != {inferred_model_config.input_dim}"
    )

    head = MILPFAttnHead(
        config=MILPFAttnHeadConfig(
            model=inferred_model_config,
            head_path=str(HEAD_WEIGHTS_PATH),
            strict_load=False,
            device="cpu",
            heatmap_shape=(64, 64),
        )
    )
    structured_embeddings = StructuredEmbeddings(
        embeddings=embeddings,
        instance_types=preprocessed.instance_types,
        groups=preprocessed.groups,
    )
    predictions = head.predict(structured_embeddings)
    print(
        "MILPFAttn input structured embeddings: "
        f"embeddings={tuple(structured_embeddings.embeddings.embeddings.shape)} "
        f"instance_types={tuple(structured_embeddings.instance_types.shape)} "
        f"groups={tuple(structured_embeddings.groups.shape)}"
    )
    print(f"MILPFAttn output suspicious shape: {tuple(predictions.suspicious.shape)}")
    print(f"MILPFAttn output suspicious dtype: {predictions.suspicious.dtype}")
    print(f"MILPFAttn output suspicious values: {predictions.suspicious.astype(np.int32).tolist()}")
    print(f"MILPFAttn output heatmap shape: {tuple(predictions.heatmap.shape)}")
    print(f"MILPFAttn output heatmap dtype: {predictions.heatmap.dtype}")
    print(
        f"MILPFAttn output heatmap stats: "
        f"min={float(predictions.heatmap.min())}, max={float(predictions.heatmap.max())}"
    )

    print("\n========== FINAL SUMMARY ==========")
    print(f"Cases: {len(viewset_list.viewsets)}")
    print(f"Images per case: {[len(v.images) for v in viewset_list.viewsets]}")
    print(f"Preprocessed batch size: {pre_images.shape[0]}")
    print(f"Embedding dim: {emb.shape[1]}")
    print(f"Final predictions batch size: {predictions.suspicious.shape[0]}")
    expected_case_batch = int(torch.unique(preprocessed.groups).numel())
    print(f"Expected case-level prediction batch size: {expected_case_batch}")

    assert predictions.suspicious.shape[0] == expected_case_batch
    assert predictions.heatmap.shape[0] == expected_case_batch
    assert predictions.heatmap.shape[1:] == (64, 64)
