from huggingface_hub import InferenceClient
import os

client = InferenceClient(provider="fal-ai", api_key=os.environ.get("HF_TOKEN"))

with open("data/images/raw/Gemini_Generated_Image_uhwt5guhwt5guhwt.png", "rb") as f:
    image_bytes = f.read()

print("Testing Qwen without mask (Ctrl+C to stop after 10 seconds)...")
import signal

def timeout_handler(signum, frame):
    print("\nTimeout - model not responding")
    exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(10)

try:
    result = client.image_to_image(
        image_bytes,
        prompt="Replace the object with a Coke can",
        model="Qwen/Qwen-Image-Edit-2511"
    )
    signal.alarm(0)
    print(f"Success! Got: {type(result)}")
except Exception as e:
    signal.alarm(0)
    print(f"Error: {type(e).__name__}: {str(e)[:200]}")
