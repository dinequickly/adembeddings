from huggingface_hub import InferenceClient
import os

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

client = InferenceClient(provider="fal-ai", api_key=os.environ.get("HF_TOKEN"))

with open("data/images/raw/Gemini_Generated_Image_uhwt5guhwt5guhwt.png", "rb") as f:
    image_bytes = f.read()

print("Testing Qwen model with debug logging...")
print(f"Image size: {len(image_bytes)} bytes\n")

try:
    result = client.image_to_image(
        image_bytes,
        prompt="test",
        model="Qwen/Qwen-Image-Edit-2511"
    )
    print(f"Success! Got: {type(result)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
