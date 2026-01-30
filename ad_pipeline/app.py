import json
import os
import streamlit as st
from PIL import Image, ImageDraw

from src.config import (
    RAW_DIR,
    MASK_DIR,
    VARIANT_DIR,
    DEFAULT_PROMPT,
    REVIEW_STATE_PATH,
)
from src.brief import default_brief, validate_brief
from src.io import list_images, load_json, save_json, load_review_state, save_review_state
from src.pipeline import segment_images, generate_variants, score_acceptability, image_id_from_path
from src.generate_qwen import qwen_available
from src.export import export_approved_csv


st.set_page_config(page_title="Ad Variant Review", layout="wide")


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


def _status_for_image(image_id, brands, reviews):
    mask_path = os.path.join(MASK_DIR, f"{image_id}_mask.png")
    has_mask = os.path.exists(mask_path)
    has_variant = True
    has_score = True
    for brand in brands:
        brand_name = brand["name"]
        var_path = os.path.join(VARIANT_DIR, f"{image_id}_{brand_name}.png")
        score_path = os.path.join(VARIANT_DIR, f"{image_id}_{brand_name}_score.json")
        has_variant = has_variant and os.path.exists(var_path)
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
            results = segment_images(prompt, checkpoint_path=os.environ.get("SAM3_CHECKPOINT"))
            st.write(results)

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

if brief is None:
    st.stop()

images = list_images(RAW_DIR)
if not images:
    st.info("No images found in data/images/raw")
    st.stop()

state = load_review_state()
reviews = state.get("reviews", {})

image_options = []
for path in images:
    img_id = image_id_from_path(path)
    status = _status_for_image(img_id, brief["brands"], reviews)
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
    st.image(Image.open(selected_path).convert("RGB"), caption="Original", use_column_width=True)

with mid_left:
    st.image(_mask_overlay(selected_path, mask_path), caption="Mask overlay", use_column_width=True)

brand_a = brand_names[0]
brand_b = brand_names[1] if len(brand_names) > 1 else brand_names[0]

var_a = os.path.join(VARIANT_DIR, f"{selected}_{brand_a}.png")
var_b = os.path.join(VARIANT_DIR, f"{selected}_{brand_b}.png")

with mid_right:
    if os.path.exists(var_a):
        st.image(Image.open(var_a).convert("RGB"), caption=f"{brand_a} variant", use_column_width=True)
    else:
        st.write("Missing variant")

with right:
    if os.path.exists(var_b):
        st.image(Image.open(var_b).convert("RGB"), caption=f"{brand_b} variant", use_column_width=True)
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
