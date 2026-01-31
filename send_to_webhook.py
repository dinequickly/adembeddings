import requests
import os
from ad_pipeline.src.config import RAW_DIR, MASK_DIR, ROOT_DIR
from ad_pipeline.src.brief import default_brief
from ad_pipeline.src.pipeline import image_id_from_path
from ad_pipeline.src.io import list_images

WEBHOOK_URL = "https://maxipad.app.n8n.cloud/webhook/2836304f-21a9-43b0-9afd-586fba803fa2"
BRAND_ASSETS_DIR = os.path.join(ROOT_DIR, "data", "brand_assets")

def send_images_to_webhook():
    images = list_images(RAW_DIR)
    brief = default_brief()

    for img_path in images:
        img_id = image_id_from_path(img_path)
        mask_path = os.path.join(MASK_DIR, f"{img_id}_mask.png")

        # Send one request per brand
        for brand in brief["brands"]:
            brand_name = brand["name"]
            # Use brand assets from data/brand_assets directory
            product_path = os.path.join(BRAND_ASSETS_DIR, f"{brand_name.lower()}.png")

            files = {}
            data = {"image_id": img_id, "brand": brand_name}

            # Add original image
            with open(img_path, "rb") as f:
                files["original"] = ("original.png", f.read(), "image/png")

            # Add mask image
            if os.path.exists(mask_path):
                with open(mask_path, "rb") as f:
                    files["mask"] = ("mask.png", f.read(), "image/png")

            # Add brand reference image
            if os.path.exists(product_path):
                with open(product_path, "rb") as f:
                    files["reference"] = (f"{brand_name}_reference.png", f.read(), "image/png")
                print(f"  Added reference: {product_path}")
            else:
                print(f"  Warning: Reference not found at {product_path}")

            try:
                response = requests.post(WEBHOOK_URL, data=data, files=files, timeout=300)
                print(f"✓ {img_id} - {brand_name}: {response.status_code} (files: {list(files.keys())})")
                if response.status_code >= 400:
                    print(f"  Response: {response.text[:200]}")
            except Exception as e:
                print(f"✗ {img_id} - {brand_name}: {e}")

if __name__ == "__main__":
    send_images_to_webhook()
