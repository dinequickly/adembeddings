import json
import os
from PIL import Image

from .config import SUPPORTED_IMAGE_EXTS, REVIEW_STATE_PATH


def list_images(directory):
    items = []
    if not os.path.isdir(directory):
        return items
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(SUPPORTED_IMAGE_EXTS):
            items.append(os.path.join(directory, name))
    return items


def load_image(path):
    return Image.open(path).convert("RGBA")


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_review_state(path=REVIEW_STATE_PATH):
    return load_json(path, default={"reviews": {}})


def save_review_state(state, path=REVIEW_STATE_PATH):
    save_json(path, state)
