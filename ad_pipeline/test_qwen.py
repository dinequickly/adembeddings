from huggingface_hub import InferenceClient
import os
import inspect

token = os.environ.get("HF_TOKEN")
print(f"HF_TOKEN set: {bool(token)}")

if not token:
    print("Error: HF_TOKEN not set!")
    exit(1)

client = InferenceClient(provider="fal-ai", api_key=token)

# Check what parameters image_to_image accepts
sig = inspect.signature(client.image_to_image)
print(f"\nimage_to_image signature: {sig}")
print(f"Parameters: {list(sig.parameters.keys())}")

# Load a small test image
with open("data/images/raw/Gemini_Generated_Image_uhwt5guhwt5guhwt.png", "rb") as f:
    image_bytes = f.read()
    print(f"\nImage loaded: {len(image_bytes)} bytes")

# Try calling with just the required params
print("\nAttempting call with: image, prompt, model")
print("(Press Ctrl+C if it hangs)")

try:
    result = client.image_to_image(
        image_bytes,
        prompt="test edit",
        model="Qwen/Qwen-Image-Edit-2511"
    )
    print(f"Success! Result type: {type(result)}")
except KeyboardInterrupt:
    print("\nInterrupted!")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
