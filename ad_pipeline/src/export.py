import csv
import os
import json

from .config import EXPORTS_DIR, MASK_DIR, VARIANT_DIR
from .io import load_review_state


def _load_score(image_id, brand_name):
    score_path = os.path.join(VARIANT_DIR, f"{image_id}_{brand_name}_score.json")
    if not os.path.exists(score_path):
        return None
    with open(score_path, "r") as f:
        data = json.load(f)
    return data.get("acceptability_score")


def export_approved_csv(campaign_id, brands):
    state = load_review_state()
    reviews = state.get("reviews", {})
    out_path = os.path.join(EXPORTS_DIR, f"{campaign_id}_approved.csv")
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_id",
                "brand",
                "variant_path",
                "mask_path",
                "acceptability_score",
                "approved",
                "notes",
            ]
        )
        for image_id, entry in reviews.items():
            for brand in brands:
                brand_name = brand["name"]
                review = entry.get(brand_name, {})
                approved = review.get("status") == "approved"
                notes = review.get("notes", "")
                variant_path = os.path.join(VARIANT_DIR, f"{image_id}_{brand_name}.png")
                mask_path = os.path.join(MASK_DIR, f"{image_id}_mask.png")
                score = _load_score(image_id, brand_name)
                writer.writerow(
                    [image_id, brand_name, variant_path, mask_path, score, approved, notes]
                )
    return out_path
