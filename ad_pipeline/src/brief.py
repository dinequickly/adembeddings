import json

from .config import DEFAULT_PROMPT, DEFAULT_MAX_VARIANTS


def default_brief():
    return {
        "campaign_id": "demo_campaign",
        "brands": [
            {
                "name": "Coke",
                "assets": {"logo_path": "", "product_path": ""},
                "prompt": "Coke can",
            },
            {
                "name": "Pepsi",
                "assets": {"logo_path": "", "product_path": ""},
                "prompt": "Pepsi can",
            },
        ],
        "editable_object_prompt": DEFAULT_PROMPT,
        "max_variants_per_image": DEFAULT_MAX_VARIANTS,
        "constraints": {"forbidden": [], "max_edit_strength": "medium"},
    }


def load_brief(path):
    with open(path, "r") as f:
        return json.load(f)


def save_brief(path, brief_data):
    with open(path, "w") as f:
        json.dump(brief_data, f, indent=2)


def validate_brief(brief_data):
    required_top = ["campaign_id", "brands", "editable_object_prompt", "max_variants_per_image", "constraints"]
    for key in required_top:
        if key not in brief_data:
            raise ValueError(f"Missing brief key: {key}")
    if not isinstance(brief_data["brands"], list) or len(brief_data["brands"]) < 1:
        raise ValueError("Brief must include at least one brand")
    for brand in brief_data["brands"]:
        if "name" not in brand:
            raise ValueError("Brand missing name")
        if "assets" not in brand or "product_path" not in brand["assets"]:
            raise ValueError("Brand missing assets.product_path")
        if "prompt" not in brand:
            raise ValueError("Brand missing prompt")
    return True
