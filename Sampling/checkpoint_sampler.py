import torch
import base64
import requests
from PIL import Image
import numpy as np
import os
from torchvision.utils import save_image

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2-vision"

# Sample from trained model (visual transformer embedding space)
def sample_clip_latent(clip_model, size=(1, 512)):
    return torch.randn(size)  # Or use trained latent vectors

# Save tensor as image
def tensor_to_image(tensor, save_path="sample.png"):
    tensor = tensor.squeeze(0).detach().cpu()
    save_image(tensor, save_path)
    return save_path

# Convert image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Use llama3.2-vision as discriminator
def run_ollama_discriminator(image_path):
    encoded = encode_image_to_base64(image_path)
    prompt = "Does this image resemble a human or a person? Answer yes or no."

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [encoded]
        }]
    }

    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()["message"]["content"]
    print(f"Ollama response: {result}")
    return "yes" in result.lower()

# Loop and filter
def sample_and_filter(clip_model, out_dir="samples", max_attempts=50):
    os.makedirs(out_dir, exist_ok=True)
    accepted = 0

    for i in range(max_attempts):
        latent = sample_clip_latent(clip_model)
        img_tensor = latent.view(1, 1, 32, 16)  # Mock reshape; adapt to your decoder

        # Convert to image and save
        image_path = os.path.join(out_dir, f"sample_{i}.png")
        tensor_to_image(img_tensor, image_path)

        if run_ollama_discriminator(image_path):
            print(f"Accepted: sample_{i}.png")
            accepted += 1
        else:
            os.remove(image_path)  # Optional cleanup

    print(f"Total accepted: {accepted} / {max_attempts}")
