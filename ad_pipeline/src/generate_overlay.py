import os

import numpy as np
from PIL import Image

from .io import load_image, save_image


def _mask_bbox(mask_np):
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def overlay_variant(image_path, mask_path, product_path, out_path):
    if not os.path.exists(mask_path):
        return {"status": "skipped", "reason": "mask not found"}
    if not os.path.exists(product_path):
        return {"status": "skipped", "reason": "product image not found"}

    base = load_image(image_path)
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)
    bbox = _mask_bbox(mask_np)
    if bbox is None:
        return {"status": "skipped", "reason": "empty mask"}

    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    product = Image.open(product_path).convert("RGBA")
    # Resize while preserving aspect ratio
    pw, ph = product.size
    scale = min(bw / float(pw), bh / float(ph))
    new_w, new_h = max(1, int(pw * scale)), max(1, int(ph * scale))
    product = product.resize((new_w, new_h), resample=Image.BICUBIC)

    # Center within bbox
    px = x0 + (bw - new_w) // 2
    py = y0 + (bh - new_h) // 2

    # Apply mask to product alpha
    mask_crop = mask.crop((px, py, px + new_w, py + new_h)).resize((new_w, new_h))
    product_np = np.array(product)
    alpha = product_np[:, :, 3].astype(np.float32) / 255.0
    mask_alpha = np.array(mask_crop).astype(np.float32) / 255.0
    alpha = alpha * mask_alpha
    product_np[:, :, 3] = (alpha * 255).astype(np.uint8)
    product = Image.fromarray(product_np)

    composed = base.copy()
    composed.paste(product, (px, py), product)

    save_image(composed, out_path)
    return {"status": "ok", "variant_path": out_path, "bbox": [x0, y0, x1, y1]}
