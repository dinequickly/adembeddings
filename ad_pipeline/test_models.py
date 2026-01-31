from huggingface_hub import InferenceClient
import os

client = InferenceClient(provider="fal-ai", api_key=os.environ.get("HF_TOKEN"))

# Try some alternative image editing models
models_to_try = [
    "Qwen/Qwen-Image-Edit-2511",  # Original
    "qwen-image-edit",  # Simplified name
    "fal-ai/qwen-image-edit-plus",  # FAL AI wrapper
]

with open("data/images/raw/Gemini_Generated_Image_uhwt5guhwt5guhwt.png", "rb") as f:
    image_bytes = f.read()

for model in models_to_try:
    print(f"\nTrying model: {model}")
    try:
        # Set a very short timeout by not waiting
        result = client.image_to_image(
            image_bytes,
            prompt="test",
            model=model
        )
        print(f"  ✓ Success: {type(result)}")
    except ValueError as e:
        if "not supported" in str(e):
            print(f"  ✗ Not supported: {e}")
        else:
            print(f"  ✗ ValueError: {e}")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {str(e)[:100]}")
