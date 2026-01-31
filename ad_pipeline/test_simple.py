from huggingface_hub import InferenceClient
import os

client = InferenceClient(provider="fal-ai", api_key=os.environ.get("HF_TOKEN"))

with open("data/images/raw/Gemini_Generated_Image_uhwt5guhwt5guhwt.png", "rb") as f:
    image_bytes = f.read()

print("Testing with a simpler model...")
try:
    result = client.image_to_image(
        image_bytes,
        prompt="make it more colorful",
        model="black-forest-labs/FLUX.1-schnell"
    )
    print(f"Success! Got: {type(result)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
