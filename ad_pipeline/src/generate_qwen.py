import os
import inspect

from huggingface_hub import InferenceClient
from PIL import Image

from .config import QWEN_MODEL_ID, QWEN_PROVIDER
from .io import save_image


def qwen_available():
    return bool(os.environ.get("HF_TOKEN"))


def generate_qwen_variant(image_path, mask_path, prompt, out_path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"status": "skipped", "reason": "HF_TOKEN not set"}

    client = InferenceClient(provider=QWEN_PROVIDER, token=token)

    image = Image.open(image_path).convert("RGB")
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")

    try:
        # Try passing mask if supported
        sig = inspect.signature(client.image_to_image)
        kwargs = {"image": image, "prompt": prompt, "model": QWEN_MODEL_ID}
        if mask is not None and "mask" in sig.parameters:
            kwargs["mask"] = mask
        result = client.image_to_image(**kwargs)
    except Exception as e:
        # Fallback: call without mask if mask unsupported
        try:
            result = client.image_to_image(image=image, prompt=prompt, model=QWEN_MODEL_ID)
        except Exception as e2:
            return {"status": "error", "reason": f"Qwen edit failed: {e2}"}

    if isinstance(result, Image.Image):
        save_image(result.convert("RGBA"), out_path)
        return {"status": "ok", "variant_path": out_path, "used_mask": mask is not None}

    # Some clients return bytes
    if hasattr(result, "read"):
        img = Image.open(result).convert("RGBA")
        save_image(img, out_path)
        return {"status": "ok", "variant_path": out_path, "used_mask": mask is not None}

    return {"status": "error", "reason": "Unsupported Qwen response type"}
