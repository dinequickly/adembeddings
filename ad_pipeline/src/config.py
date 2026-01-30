import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "images", "raw")
MASK_DIR = os.path.join(DATA_DIR, "images", "masks")
VARIANT_DIR = os.path.join(DATA_DIR, "images", "variants")
REVIEWS_DIR = os.path.join(DATA_DIR, "reviews")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")

DEFAULT_PROMPT = "can"
DEFAULT_MAX_VARIANTS = 2

REVIEW_STATE_PATH = os.path.join(REVIEWS_DIR, "reviews.json")

QWEN_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
QWEN_PROVIDER = "fal-ai"

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg")
