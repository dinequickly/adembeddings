import os
import numpy as np
from PIL import Image


def _crop_to_mask(image, mask):
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image, None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return image.crop((x0, y0, x1 + 1, y1 + 1)), (x0, y0, x1, y1)


def _hist_similarity(img_a, img_b):
    a = np.array(img_a.resize((64, 64))).reshape(-1, 3).astype(np.float32)
    b = np.array(img_b.resize((64, 64))).reshape(-1, 3).astype(np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
    sim = (a * b).sum(axis=1).mean()
    return float(sim)


def _artifact_score(img):
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    var = float(arr.var())
    score = min(1.0, var * 5.0)
    return score


def _ocr_brand_score(img, brand_name):
    try:
        import pytesseract
    except Exception:
        return None
    text = pytesseract.image_to_string(img)
    if brand_name.lower() in text.lower():
        return 1.0
    return 0.2


def compute_acceptability(
    variant_path,
    mask_path,
    brand_name,
    brand_reference_path=None,
):
    variant = Image.open(variant_path).convert("RGB")
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        crop, _ = _crop_to_mask(variant, mask)
    else:
        crop = variant

    # Prefer OCR if available
    ocr_score = _ocr_brand_score(crop, brand_name)
    if ocr_score is not None:
        brand_score = ocr_score
        brand_method = "ocr"
    elif brand_reference_path and os.path.exists(brand_reference_path):
        ref = Image.open(brand_reference_path).convert("RGB")
        brand_score = _hist_similarity(crop, ref)
        brand_method = "hist"
    else:
        brand_score = 0.5
        brand_method = "fallback"

    art_score = _artifact_score(crop)
    acceptability = 0.6 * brand_score + 0.4 * art_score

    return {
        "acceptability_score": float(acceptability),
        "brand_score": float(brand_score),
        "artifact_score": float(art_score),
        "brand_method": brand_method,
    }
