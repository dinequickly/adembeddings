import json
import os
import inspect

import numpy as np
from PIL import Image

from .config import MASK_DIR


def _try_import_sam3():
    errors = []
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        return Sam3Processor, build_sam3_image_model, None
    except Exception as e:
        errors.append(str(e))
    return None, None, " | ".join(errors)


def _build_model(build_fn, checkpoint_path=None):
    kwargs = {}
    if checkpoint_path:
        try:
            sig = inspect.signature(build_fn)
            for name in ["checkpoint", "checkpoint_path", "ckpt_path", "ckpt"]:
                if name in sig.parameters:
                    kwargs[name] = checkpoint_path
                    break
        except Exception:
            pass
    return build_fn(**kwargs)


def _build_processor(processor_cls, model):
    try:
        sig = inspect.signature(processor_cls)
        if "model" in sig.parameters:
            return processor_cls(model=model)
    except Exception:
        pass
    return processor_cls(model)


def _run_processor(processor, image, prompt):
    # Official repo usage: set_image -> set_text_prompt(state=..., prompt=...)
    if hasattr(processor, "set_image") and hasattr(processor, "set_text_prompt"):
        inference_state = processor.set_image(image)
        return processor.set_text_prompt(state=inference_state, prompt=prompt)
    # Fallback for unexpected API variants
    if hasattr(processor, "set_text_prompt"):
        return processor.set_text_prompt(prompt=prompt)
    raise RuntimeError("SAM3 processor API mismatch; expected set_image + set_text_prompt")


def _extract_masks(result):
    masks = None
    scores = None
    boxes = None
    if isinstance(result, dict):
        masks = result.get("masks")
        scores = result.get("scores")
        boxes = result.get("boxes")
    elif isinstance(result, (list, tuple)):
        if len(result) >= 1:
            masks = result[0]
        if len(result) >= 2:
            scores = result[1]
        if len(result) >= 3:
            boxes = result[2]
    if masks is None:
        raise RuntimeError("SAM3 output did not include masks")
    if scores is None:
        scores = [1.0] * len(masks)
    return masks, scores, boxes


def _to_numpy_mask(mask):
    if isinstance(mask, np.ndarray):
        return mask
    try:
        import torch

        if isinstance(mask, torch.Tensor):
            return mask.detach().cpu().numpy()
    except Exception:
        pass
    return np.array(mask)


def _save_mask(mask_np, out_path):
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    Image.fromarray(mask_np).save(out_path)


def segment_image(
    image_path,
    prompt,
    checkpoint_path=None,
    out_mask_path=None,
    out_meta_path=None,
    allow_fallback=False,
):
    out_mask_path = out_mask_path or os.path.join(
        MASK_DIR, os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
    )
    out_meta_path = out_meta_path or os.path.splitext(out_mask_path)[0] + ".json"

    sam3_processor_cls, build_fn, import_err = _try_import_sam3()
    if sam3_processor_cls is None or build_fn is None:
        msg = "SAM3 not available. Install the official SAM3 repo and ensure it is on PYTHONPATH. "
        msg += f"Import errors: {import_err}"
        if allow_fallback:
            image = Image.open(image_path).convert("L")
            w, h = image.size
            mask = np.zeros((h, w), dtype=np.uint8)
            x0, y0 = w // 4, h // 4
            x1, y1 = 3 * w // 4, 3 * h // 4
            mask[y0:y1, x0:x1] = 255
            _save_mask(mask, out_mask_path)
            meta = {"status": "fallback", "reason": msg, "score": 0.0, "box": [x0, y0, x1, y1]}
            with open(out_meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            return {"status": "fallback", "mask_path": out_mask_path, "meta_path": out_meta_path}
        return {"status": "skipped", "reason": msg}

    try:
        import torch

        if not torch.cuda.is_available():
            print("Warning: CUDA not available. SAM3 may be slow or fail on CPU.")
    except Exception:
        print("Warning: torch not available; cannot check CUDA for SAM3.")

    model = _build_model(build_fn, checkpoint_path=checkpoint_path)
    processor = _build_processor(sam3_processor_cls, model)
    image = Image.open(image_path).convert("RGB")
    result = _run_processor(processor, image, prompt)
    masks, scores, boxes = _extract_masks(result)

    # Pick top mask by score
    scores_np = np.array(scores)
    top_idx = int(np.argmax(scores_np))
    mask_np = _to_numpy_mask(masks[top_idx])
    box = None
    if boxes is not None and len(boxes) > top_idx:
        box = boxes[top_idx]

    _save_mask(mask_np, out_mask_path)
    meta = {"status": "ok", "score": float(scores_np[top_idx]), "box": box}
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"status": "ok", "mask_path": out_mask_path, "meta_path": out_meta_path}


def segment_folder(image_paths, prompt, checkpoint_path=None, allow_fallback=False):
    results = []
    for path in image_paths:
        res = segment_image(
            path,
            prompt,
            checkpoint_path=checkpoint_path,
            allow_fallback=allow_fallback,
        )
        results.append(res)
    return results
