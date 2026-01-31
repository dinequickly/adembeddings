import json
import os
import time
import streamlit as st
from PIL import Image, ImageDraw

from src.config import (
    RAW_DIR,
    MASK_DIR,
    VARIANT_DIR,
    DEFAULT_PROMPT,
    REVIEW_STATE_PATH,
    ROOT_DIR,
    SUPPORTED_IMAGE_EXTS,
)
from src.brief import default_brief, validate_brief
from src.io import list_images, load_json, save_json, load_review_state, save_review_state
from src.pipeline import segment_images, generate_variants, score_acceptability, image_id_from_path
from src.generate_qwen import qwen_available
from src.export import export_approved_csv
import requests


st.set_page_config(page_title="Ad Variant Review", layout="wide")

N8N_RESULTS_PATH = os.path.join(ROOT_DIR, "data", "n8n_results.json")


def _mask_overlay(image_path, mask_path):
    img = Image.open(image_path).convert("RGBA")
    if not os.path.exists(mask_path):
        return img
    mask = Image.open(mask_path).convert("L")
    overlay = Image.new("RGBA", img.size, (255, 0, 0, 80))
    img.paste(overlay, (0, 0), mask)
    return img


def _load_scores(image_id, brands):
    scores = {}
    for brand in brands:
        brand_name = brand["name"]
        score_path = os.path.join(VARIANT_DIR, f"{image_id}_{brand_name}_score.json")
        if os.path.exists(score_path):
            scores[brand_name] = load_json(score_path, default={})
    return scores


def _load_webhook_results():
    try:
        data = load_json(N8N_RESULTS_PATH, default={}) or {}
    except Exception:
        data = {}
    results = {}
    for key, value in data.items():
        if isinstance(value, dict):
            url = value.get("image_url") or value.get("url") or value.get("link")
        else:
            url = value
        if url:
            results[key] = url
    return results


def _slugify_name(value):
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _resolve_brand_product_path(brand, brand_assets_dir):
    assets = brand.get("assets", {}) if isinstance(brand, dict) else {}
    product_path = assets.get("product_path") if isinstance(assets, dict) else None

    if product_path:
        candidate = product_path
        if not os.path.isabs(candidate):
            candidate = os.path.join(ROOT_DIR, candidate)
        if os.path.exists(candidate):
            return candidate

    if not os.path.isdir(brand_assets_dir):
        return None

    brand_name = (brand.get("name") if isinstance(brand, dict) else "") or ""
    brand_slug = _slugify_name(brand_name.strip())
    if not brand_slug:
        return None

    best_path = None
    best_score = -1.0
    for filename in os.listdir(brand_assets_dir):
        if not filename.lower().endswith(SUPPORTED_IMAGE_EXTS):
            continue
        stem = os.path.splitext(filename)[0]
        stem_slug = _slugify_name(stem)
        if not stem_slug:
            continue
        score = -1.0
        if brand_slug == stem_slug:
            score = 100.0
        elif brand_slug.startswith(stem_slug):
            score = 80.0 + len(stem_slug) / 100.0
        elif stem_slug.startswith(brand_slug):
            score = 70.0 + len(brand_slug) / 100.0
        elif brand_slug in stem_slug or stem_slug in brand_slug:
            score = 60.0 + min(len(brand_slug), len(stem_slug)) / 100.0
        if score > best_score:
            best_score = score
            best_path = os.path.join(brand_assets_dir, filename)

    return best_path


def _send_to_webhook(brief, webhook_url):
    """Send images to webhook for each brand"""
    images = list_images(RAW_DIR)
    brand_assets_dir = os.path.join(ROOT_DIR, "data", "brand_assets")
    results = []

    if not webhook_url:
        return [{"error": "Webhook URL is empty"}]

    for img_path in images:
        img_id = image_id_from_path(img_path)
        mask_path = os.path.join(MASK_DIR, f"{img_id}_mask.png")

        for brand in brief["brands"]:
            brand_name = brand["name"]
            product_path = _resolve_brand_product_path(brand, brand_assets_dir)

            files = {}
            data = {"image_id": img_id, "brand": brand_name}

            try:
                with open(img_path, "rb") as f:
                    files["original"] = ("original.png", f.read(), "image/png")

                if os.path.exists(mask_path):
                    with open(mask_path, "rb") as f:
                        files["mask"] = ("mask.png", f.read(), "image/png")

                if product_path and os.path.exists(product_path):
                    with open(product_path, "rb") as f:
                        files["reference"] = (f"{brand_name}_reference.png", f.read(), "image/png")

                response = requests.post(webhook_url, data=data, files=files, timeout=300)

                result = {
                    "image_id": img_id,
                    "brand": brand_name,
                    "status": response.status_code,
                }

                if response.status_code == 200:
                    try:
                        resp_json = response.json()
                        # Try different possible field names for the URL
                        result["image_url"] = resp_json.get("link") or resp_json.get("data") or resp_json.get("url") or ""
                    except:
                        result["image_url"] = response.text
                else:
                    result["error"] = response.text[:200]

                results.append(result)
            except Exception as e:
                results.append({
                    "image_id": img_id,
                    "brand": brand_name,
                    "status": "error",
                    "error": str(e)
                })

    return results


def _status_for_image(image_id, brands, reviews, webhook_results=None):
    mask_path = os.path.join(MASK_DIR, f"{image_id}_mask.png")
    has_mask = os.path.exists(mask_path)
    has_variant = True
    has_score = True
    webhook_results = webhook_results or {}
    for brand in brands:
        brand_name = brand["name"]
        var_path = os.path.join(VARIANT_DIR, f"{image_id}_{brand_name}.png")
        score_path = os.path.join(VARIANT_DIR, f"{image_id}_{brand_name}_score.json")
        webhook_key = f"{image_id}_{brand_name}"
        has_variant = has_variant and (os.path.exists(var_path) or webhook_key in webhook_results)
        has_score = has_score and os.path.exists(score_path)
    if not has_mask:
        return "raw"
    if not has_variant:
        return "masked"
    if not has_score:
        return "generated"
    if image_id in reviews:
        return "reviewed"
    return "scored"


st.title("Image Ad Variant Pipeline")

with st.sidebar:
    st.header("Campaign Brief")
    brief_json = st.text_area("Brief JSON", value=json.dumps(default_brief(), indent=2), height=300)
    brief = None
    try:
        brief = json.loads(brief_json)
        validate_brief(brief)
        if brief:
            brand_assets_dir = os.path.join(ROOT_DIR, "data", "brand_assets")
            for brand in brief.get("brands", []):
                assets = brand.get("assets") if isinstance(brand, dict) else None
                if not isinstance(assets, dict):
                    assets = {}
                    brand["assets"] = assets
                current_path = assets.get("product_path")
                if not current_path or not os.path.exists(
                    current_path if os.path.isabs(current_path) else os.path.join(ROOT_DIR, current_path)
                ):
                    resolved = _resolve_brand_product_path(brand, brand_assets_dir)
                    if resolved:
                        assets["product_path"] = resolved
        st.success("Brief OK")
    except Exception as e:
        st.error(f"Brief error: {e}")

    prompt = st.text_input("SAM prompt", value=DEFAULT_PROMPT)
    backend = st.selectbox(
        "Generation backend",
        ["overlay", "qwen"],
        index=0,
        disabled=not qwen_available(),
    )
    if not qwen_available():
        st.caption("HF_TOKEN missing; Qwen backend disabled")

    if st.button("Run segmentation"):
        if brief:
            images = list_images(RAW_DIR)
            existing_masks = []
            missing_images = []
            for img_path in images:
                img_id = image_id_from_path(img_path)
                mask_path = os.path.join(MASK_DIR, f"{img_id}_mask.png")
                if os.path.exists(mask_path):
                    existing_masks.append((img_id, mask_path))
                else:
                    missing_images.append(img_path)

            if existing_masks:
                st.subheader("Existing masks")
                for img_id, mask_path in existing_masks:
                    st.image(Image.open(mask_path).convert("L"), caption=f"{img_id} mask (existing)")

            if missing_images:
                results = segment_images(
                    prompt,
                    checkpoint_path=os.environ.get("SAM3_CHECKPOINT"),
                    image_paths=missing_images,
                )
                st.write(results)
            else:
                st.info("All masks already exist; skipping segmentation.")

    if st.button("Generate variants"):
        if brief:
            results = generate_variants(brief, backend=backend)
            st.write(results)

    if st.button("Score acceptability"):
        if brief:
            results = score_acceptability(brief)
            st.write(results)

    if st.button("Export approved CSV"):
        if brief:
            out_path = export_approved_csv(brief["campaign_id"], brief["brands"])
            st.success(f"Exported: {out_path}")

    st.divider()
    st.subheader("Webhook")
    webhook_url = st.text_input(
        "Webhook URL",
        value="https://maxipad.app.n8n.cloud/webhook/2836304f-21a9-43b0-9afd-586fba803fa2",
        type="password"
    )
    if st.button("Send to webhook"):
        if brief and webhook_url:
            with st.spinner("Sending to N8N..."):
                results = _send_to_webhook(brief, webhook_url)

                # Save to file for Next.js feed
                n8n_data = {}

                # Load existing results if file exists
                if os.path.exists(N8N_RESULTS_PATH):
                    try:
                        with open(N8N_RESULTS_PATH, 'r') as f:
                            n8n_data = json.load(f)
                    except:
                        n8n_data = {}

                # Add new results
                for result in results:
                    if result.get("status") == 200 and result.get("image_url"):
                        key = f"{result['image_id']}_{result['brand']}"
                        n8n_data[key] = {
                            "image_url": result["image_url"],
                            "timestamp": int(time.time()),
                            "image_id": result["image_id"],
                            "brand": result["brand"]
                        }

                # Save to file
                with open(N8N_RESULTS_PATH, 'w') as f:
                    json.dump(n8n_data, f, indent=2)

                # Also keep in session state for Streamlit UI
                if "webhook_results" not in st.session_state:
                    st.session_state.webhook_results = {}
                for result in results:
                    if result.get("status") == 200 and result.get("image_url"):
                        key = f"{result['image_id']}_{result['brand']}"
                        st.session_state.webhook_results[key] = result["image_url"]

                st.success(f"Generated {len([r for r in results if r.get('status') == 200])} images")

if brief is None:
    st.stop()

images = list_images(RAW_DIR)
if not images:
    st.info("No images found in data/images/raw")
    st.stop()

state = load_review_state()
reviews = state.get("reviews", {})
file_webhook_results = _load_webhook_results()
session_webhook_results = st.session_state.get("webhook_results", {})
webhook_results = {**file_webhook_results, **session_webhook_results}

image_options = []
for path in images:
    img_id = image_id_from_path(path)
    status = _status_for_image(img_id, brief["brands"], reviews, webhook_results=webhook_results)
    image_options.append((img_id, status, path))

st.subheader("Images")
for img_id, status, _ in image_options:
    st.write(f"{img_id}: {status}")

selected = st.selectbox("Select image", [x[0] for x in image_options])
selected_path = None
for img_id, _, path in image_options:
    if img_id == selected:
        selected_path = path
        break

if not selected_path:
    st.stop()

mask_path = os.path.join(MASK_DIR, f"{selected}_mask.png")

brand_names = [b["name"] for b in brief["brands"]]
if len(brand_names) < 2:
    st.warning("Brief must contain at least two brands for comparison")

left, mid_left, mid_right, right = st.columns(4)

with left:
    st.image(Image.open(selected_path).convert("RGB"), caption="Original")

with mid_left:
    st.image(_mask_overlay(selected_path, mask_path), caption="Mask overlay")

brand_a = brand_names[0]
brand_b = brand_names[1] if len(brand_names) > 1 else brand_names[0]

# Check if N8N webhook results exist, otherwise use local variants
var_a_key = f"{selected}_{brand_a}"
var_b_key = f"{selected}_{brand_b}"

with mid_right:
    if var_a_key in webhook_results:
        st.image(webhook_results[var_a_key], caption=f"{brand_a} variant (N8N)")
    else:
        var_a = os.path.join(VARIANT_DIR, f"{selected}_{brand_a}.png")
        if os.path.exists(var_a):
            st.image(Image.open(var_a).convert("RGB"), caption=f"{brand_a} variant")
        else:
            st.write("Missing variant")

with right:
    if var_b_key in webhook_results:
        st.image(webhook_results[var_b_key], caption=f"{brand_b} variant (N8N)")
    else:
        var_b = os.path.join(VARIANT_DIR, f"{selected}_{brand_b}.png")
        if os.path.exists(var_b):
            st.image(Image.open(var_b).convert("RGB"), caption=f"{brand_b} variant")
        else:
            st.write("Missing variant")

scores = _load_scores(selected, brief["brands"])
if scores:
    st.subheader("Acceptability scores")
    for brand_name, data in scores.items():
        st.write(f"{brand_name}: {data.get('acceptability_score'):.3f} (method={data.get('brand_method')})")

entry = reviews.get(selected, {})

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button(f"Approve {brand_a}"):
        entry[brand_a] = {"status": "approved", "notes": ""}
        reviews[selected] = entry
        state["reviews"] = reviews
        save_review_state(state, REVIEW_STATE_PATH)
        st.success(f"Approved {brand_a}")

with col2:
    if st.button(f"Approve {brand_b}"):
        entry[brand_b] = {"status": "approved", "notes": ""}
        reviews[selected] = entry
        state["reviews"] = reviews
        save_review_state(state, REVIEW_STATE_PATH)
        st.success(f"Approved {brand_b}")

with col3:
    if st.button("Reject both"):
        for b in brand_names:
            entry[b] = {"status": "rejected", "notes": ""}
        reviews[selected] = entry
        state["reviews"] = reviews
        save_review_state(state, REVIEW_STATE_PATH)
        st.warning("Rejected both")

with col4:
    if st.button("Needs manual fix"):
        for b in brand_names:
            entry[b] = {"status": "needs_manual_fix", "notes": ""}
        reviews[selected] = entry
        state["reviews"] = reviews
        save_review_state(state, REVIEW_STATE_PATH)
        st.info("Marked as needs manual fix")
