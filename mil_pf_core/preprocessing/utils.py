from typing import List, Sequence, Tuple

import cv2
import numpy as np


def _add_one_if_even(number: int) -> int:
    return number + 1 if number % 2 == 0 else number


def _create_kernel(shape: Tuple[int, int], factor: int) -> Tuple[int, int]:
    h, w = shape
    return (
        _add_one_if_even(max(1, h // factor)),
        _add_one_if_even(max(1, w // factor)),
    )


def _scale_to_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max <= img_min:
        return np.zeros_like(img, dtype=np.uint8)
    img = 255.0 * (img - img_min) / (img_max - img_min)
    return img.astype(np.uint8)


def binarize(image: np.ndarray) -> np.ndarray:
    img_u8 = _scale_to_uint8(image)
    blurred = cv2.GaussianBlur(img_u8, _create_kernel(img_u8.shape, 100), 0)
    _, mask = cv2.threshold(
        blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask.astype(np.uint8)


def erode(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, _create_kernel(mask.shape, 500)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)


def dilate(mask: np.ndarray, dilation_factor: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, _create_kernel(mask.shape, dilation_factor)
    )
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)


def keep_largest_blob(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.ones_like(mask, dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(out, [largest], -1, 1, thickness=cv2.FILLED)
    return out


def get_breast_mask(
    image: np.ndarray, dilation_factor: int = 10
) -> np.ndarray:
    mask = binarize(image)
    mask = erode(mask)
    mask = dilate(mask, dilation_factor)
    return keep_largest_blob(mask)


def keep_only_breast(
    image: np.ndarray, dilation_factor: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    mask = get_breast_mask(image, dilation_factor=dilation_factor)
    return image * mask, mask


def should_flip(image: np.ndarray) -> bool:
    x_center = image.shape[1] // 2
    col_sum = image.sum(axis=0)
    left_sum = np.sum(col_sum[:x_center])
    right_sum = np.sum(col_sum[x_center:])
    return left_sum < right_sum


def flip_if_should(image: np.ndarray) -> np.ndarray:
    return np.fliplr(image) if should_flip(image) else image


def pad_to_aspect_ratio(image: np.ndarray, aspect_ratio: float) -> np.ndarray:
    h, w = image.shape
    image_ratio = h / w
    if np.isclose(image_ratio, aspect_ratio):
        return image

    if aspect_ratio < image_ratio:
        new_w = int(h / aspect_ratio)
        out = np.zeros((h, new_w), dtype=image.dtype)
    else:
        new_h = int(w * aspect_ratio)
        out = np.zeros((new_h, w), dtype=image.dtype)
    out[:h, :w] = image
    return out


def otsu_cut(image: np.ndarray) -> np.ndarray:
    img_u8 = _scale_to_uint8(image)
    _, mask = cv2.threshold(
        img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if mask.size == 0 or not np.any(mask):
        return image

    rows = np.any(mask == 255, axis=1)
    cols = np.any(mask == 255, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    if r_idx.size == 0 or c_idx.size == 0:
        return image
    return image[r_idx[0] : r_idx[-1] + 1, c_idx[0] : c_idx[-1] + 1]


def resize_img(image: np.ndarray, resize: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray, scale_to_unit_interval: bool) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if not scale_to_unit_interval:
        return img
    img_min = float(img.min())
    img_max = float(img.max())
    if img_max <= img_min:
        return np.zeros_like(img, dtype=np.float32)
    return (img - img_min) / (img_max - img_min)


def negate_if_should(image: np.ndarray) -> np.ndarray:
    num_bins = 20
    img = np.asarray(image)
    hist, _ = np.histogram(
        img.ravel(), bins=num_bins, range=[img.min(), img.max()]
    )
    max_bin = int(np.argmax(hist))
    return img if max_bin < (num_bins / 2) else np.max(img) - img


def pad_images_to_max_shape(images: Sequence[np.ndarray]) -> List[np.ndarray]:
    if not images:
        return []
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        out = np.zeros((max_h, max_w), dtype=img.dtype)
        h, w = img.shape
        out[:h, :w] = img
        padded.append(out)
    return padded


def tile_single(
    image: np.ndarray,
    tile_size: Tuple[int, int] = (518, 518),
    overlap: float = 0.25,
    threshold: float = 0.05,
    tile_increase_tolerance: float = 0.1,
) -> List[np.ndarray]:
    h, w = image.shape
    base_tile_h, base_tile_w = tile_size

    mask = get_breast_mask(image)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    region_h = y_max - y_min + 1
    region_w = x_max - x_min + 1

    best_tile_h = base_tile_h
    best_tile_w = base_tile_w

    for scale_h in np.linspace(1.0, 1.0 + tile_increase_tolerance, 10):
        tile_h = int(base_tile_h * scale_h)
        num_tiles_h = max(
            1,
            int(np.ceil((region_h - tile_h * overlap) / (tile_h * (1 - overlap)))),
        )
        stride_h = (region_h - tile_h) / max(1, num_tiles_h - 1) if num_tiles_h > 1 else 0
        if int(np.floor(stride_h)) >= 0:
            best_tile_h = tile_h

    for scale_w in np.linspace(1.0, 1.0 + tile_increase_tolerance, 10):
        tile_w = int(base_tile_w * scale_w)
        num_tiles_w = max(
            1,
            int(np.ceil((region_w - tile_w * overlap) / (tile_w * (1 - overlap)))),
        )
        stride_w = (region_w - tile_w) / max(1, num_tiles_w - 1) if num_tiles_w > 1 else 0
        if int(np.floor(stride_w)) >= 0:
            best_tile_w = tile_w

    tile_h = best_tile_h
    tile_w = best_tile_w

    num_tiles_h = max(
        1, int(np.ceil((region_h - tile_h * overlap) / (tile_h * (1 - overlap))))
    )
    num_tiles_w = max(
        1, int(np.ceil((region_w - tile_w * overlap) / (tile_w * (1 - overlap))))
    )

    stride_h = (region_h - tile_h) / max(1, num_tiles_h - 1) if num_tiles_h > 1 else 0
    stride_w = (region_w - tile_w) / max(1, num_tiles_w - 1) if num_tiles_w > 1 else 0
    stride_h = int(np.floor(stride_h))
    stride_w = int(np.floor(stride_w))

    y_start = max(0, y_min)
    x_start = max(0, x_min)
    max_value = float(np.max(image)) if image.size else 0.0

    tiles: List[np.ndarray] = []
    for i in range(num_tiles_h):
        y = y_start + i * stride_h
        if y + tile_h > h:
            y = h - tile_h
        for j in range(num_tiles_w):
            x = x_start + j * stride_w
            if x + tile_w > w:
                x = w - tile_w
            tile_img = image[y : y + tile_h, x : x + tile_w]
            if max_value <= 0 or tile_img.mean() > threshold * max_value:
                tiles.append(tile_img)

    return tiles


def tile_multiple(
    images: Sequence[np.ndarray],
    tile_size: Tuple[int, int] = (518, 518),
    overlap: float = 0.25,
    threshold: float = 0.05,
    tile_increase_tolerance: float = 0.1,
) -> List[np.ndarray]:
    tiles: List[np.ndarray] = []
    for image in images:
        tiles.extend(
            tile_single(
                image=image,
                tile_size=tile_size,
                overlap=overlap,
                threshold=threshold,
                tile_increase_tolerance=tile_increase_tolerance,
            )
        )
    return tiles


def center_crop_or_pad(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    out = np.zeros((target_h, target_w), dtype=image.dtype)

    h, w = image.shape
    src_y0 = max(0, (h - target_h) // 2)
    src_x0 = max(0, (w - target_w) // 2)
    src_y1 = min(h, src_y0 + target_h)
    src_x1 = min(w, src_x0 + target_w)

    cropped = image[src_y0:src_y1, src_x0:src_x1]
    ch, cw = cropped.shape

    dst_y0 = max(0, (target_h - ch) // 2)
    dst_x0 = max(0, (target_w - cw) // 2)
    out[dst_y0 : dst_y0 + ch, dst_x0 : dst_x0 + cw] = cropped
    return out


def preprocess_single(
    image: np.ndarray,
    aspect_ratio: float,
    shape: Tuple[int, int],
    breast_mask_dilation_factor: int,
) -> np.ndarray:
    img = np.asarray(image)
    img = negate_if_should(img)
    img = flip_if_should(img)
    img, _ = keep_only_breast(img, dilation_factor=breast_mask_dilation_factor)
    img = otsu_cut(img)
    img = pad_to_aspect_ratio(img, aspect_ratio)
    img = resize_img(img, shape)
    img = normalize_image(img, scale_to_unit_interval=True)
    return img
