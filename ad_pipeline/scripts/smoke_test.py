import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config import RAW_DIR, MASK_DIR, VARIANT_DIR
from src.io import list_images
from src.brief import default_brief
from src.segment_sam3 import segment_image
from src.generate_overlay import overlay_variant
from src.acceptability import compute_acceptability


def main():
    images = list_images(RAW_DIR)
    if not images:
        print("No images found in data/images/raw")
        return

    image_path = images[0]
    img_id = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(MASK_DIR, f"{img_id}_mask.png")

    print("Running SAM3 segmentation...")
    res = segment_image(image_path, "can", checkpoint_path=os.environ.get("SAM3_CHECKPOINT"))
    if res.get("status") != "ok":
        print("SAM3 segmentation skipped or failed. Reason:")
        print(res.get("reason"))
        print("Using fallback center mask for smoke test.")
        res = segment_image(
            image_path,
            "can",
            checkpoint_path=os.environ.get("SAM3_CHECKPOINT"),
            allow_fallback=True,
        )

    brief = default_brief()
    for brand in brief["brands"]:
        brand_name = brand["name"]
        product_path = brand["assets"]["product_path"]
        if not product_path:
            print(f"Brand {brand_name} missing product_path; skip")
            continue
        out_path = os.path.join(VARIANT_DIR, f"{img_id}_{brand_name}.png")
        gen = overlay_variant(image_path, mask_path, product_path, out_path)
        print(f"Overlay result for {brand_name}: {gen}")
        if gen.get("status") == "ok":
            score = compute_acceptability(out_path, mask_path, brand_name, product_path)
            print(f"Acceptability for {brand_name}: {score}")
            print(f"Variant path: {out_path}")


if __name__ == "__main__":
    main()
