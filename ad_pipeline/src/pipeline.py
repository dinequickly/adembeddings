import os
import json

from .config import RAW_DIR, MASK_DIR, VARIANT_DIR
from .io import list_images
from .segment_sam3 import segment_folder
from .generate_overlay import overlay_variant
from .generate_qwen import generate_qwen_variant, qwen_available
from .acceptability import compute_acceptability


def image_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def segment_images(prompt, checkpoint_path=None, allow_fallback=False):
    images = list_images(RAW_DIR)
    return segment_folder(images, prompt, checkpoint_path=checkpoint_path, allow_fallback=allow_fallback)


def generate_variants(brief, backend="overlay"):
    images = list_images(RAW_DIR)
    results = []
    for img_path in images:
        img_id = image_id_from_path(img_path)
        mask_path = os.path.join(MASK_DIR, f"{img_id}_mask.png")
        for brand in brief["brands"]:
            brand_name = brand["name"]
            product_path = brand["assets"]["product_path"]
            out_path = os.path.join(VARIANT_DIR, f"{img_id}_{brand_name}.png")
            if backend == "qwen" and qwen_available():
                prompt = (
                    f"Replace the object inside the provided mask with a realistic {brand_name} can. "
                    "Keep lighting consistent."
                )
                res = generate_qwen_variant(img_path, mask_path, prompt, out_path)
            else:
                res = overlay_variant(img_path, mask_path, product_path, out_path)
            res.update({"image_id": img_id, "brand": brand_name, "variant_path": out_path})
            results.append(res)
    return results


def score_acceptability(brief):
    images = list_images(RAW_DIR)
    results = []
    for img_path in images:
        img_id = image_id_from_path(img_path)
        mask_path = os.path.join(MASK_DIR, f"{img_id}_mask.png")
        for brand in brief["brands"]:
            brand_name = brand["name"]
            ref_path = brand["assets"].get("product_path")
            variant_path = os.path.join(VARIANT_DIR, f"{img_id}_{brand_name}.png")
            if not os.path.exists(variant_path):
                continue
            score = compute_acceptability(
                variant_path=variant_path,
                mask_path=mask_path,
                brand_name=brand_name,
                brand_reference_path=ref_path,
            )
            score_path = os.path.join(VARIANT_DIR, f"{img_id}_{brand_name}_score.json")
            with open(score_path, "w") as f:
                json.dump(score, f, indent=2)
            results.append({"image_id": img_id, "brand": brand_name, **score})
    return results
