import os

from huggingface_hub import InferenceClient
from PIL import Image
import io

from .config import QWEN_MODEL_ID, QWEN_PROVIDER
from .io import save_image


def qwen_available():
    return bool(os.environ.get("HF_TOKEN"))


def _make_reference_composite(original, reference):
    # Place original on the left and reference on the right.
    ow, oh = original.size
    rw, rh = reference.size
    scale = oh / float(rh)
    ref_w = max(1, int(rw * scale))
    reference = reference.resize((ref_w, oh), resample=Image.BICUBIC)
    gap = max(10, int(0.02 * ow))
    composite = Image.new("RGB", (ow + gap + ref_w, oh), (255, 255, 255))
    composite.paste(original, (0, 0))
    composite.paste(reference, (ow + gap, 0))
    return composite


def _image_to_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _call_qwen(client, image_bytes, prompt):
    # FAL AI's Qwen model doesn't properly support mask parameter
    return client.image_to_image(image_bytes, prompt=prompt, model=QWEN_MODEL_ID)


def generate_qwen_variant(image_path, mask_path, prompt, out_path, reference_image_path=None):
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "skipped", "reason": "HF_TOKEN not set"}

    client = InferenceClient(provider=QWEN_PROVIDER, token=token)

    image = Image.open(image_path).convert("RGB")
    original_width = image.size[0]
    composite_used = False
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
    if reference_image_path and os.path.exists(reference_image_path):
        ref = Image.open(reference_image_path).convert("RGB")
        image = _make_reference_composite(image, ref)
        composite_used = True
        prompt = (
            prompt
            + " The input image is a side-by-side composite: LEFT is the scene to edit, RIGHT is the brand reference. "
            + "Only edit the LEFT side."
        )

    image_bytes = _image_to_bytes(image)

    try:
        result = _call_qwen(client, image_bytes, prompt)
    except Exception as e:
        return {"status": "error", "reason": f"Qwen edit failed: {e}"}

    if isinstance(result, Image.Image):
        img = result.convert("RGBA")
        if composite_used:
            img = img.crop((0, 0, original_width, img.size[1]))
        save_image(img, out_path)
        return {"status": "ok", "variant_path": out_path, "used_mask": mask is not None}

    # Some clients return bytes
    if hasattr(result, "read"):
        img = Image.open(result).convert("RGBA")
        if composite_used:
            img = img.crop((0, 0, original_width, img.size[1]))
        save_image(img, out_path)
        return {"status": "ok", "variant_path": out_path, "used_mask": mask is not None}

    return {"status": "error", "reason": "Unsupported Qwen response type"}
