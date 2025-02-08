# yt-dlp

import requests

API_KEY = "YOUR_YOUTUBE_API_KEY"
URL = "https://www.googleapis.com/youtube/v3/search"
stream_urls = []
params = {
    "part": "snippet",
    "type": "video",
    "eventType": "live",  # Only live streams
    "chart": "mostPopular",
    "regionCode": "US",  # Change to target a different country
    "maxResults": 50,
    "key": API_KEY
}

response = requests.get(URL, params=params)
data = response.json()

for idx, video in enumerate(data.get("items", []), 1):
    title = video["snippet"]["title"]
    video_id = video["id"]["videoId"]
    stream_urls.add(f"https://www.youtube.com/watch?v={video_id}")
    print(f"{idx}. {title} - https://www.youtube.com/watch?v={video_id}")




# Function to capture frames
def capture_stream():
    process = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    byte_frame = b""

    while True:
        byte_frame += process.stdout.read(1024)
        a = byte_frame.find(b'\xff\xd8')  # JPEG start
        b = byte_frame.find(b'\xff\xd9')  # JPEG end

        if a != -1 and b != -1:
            jpg = byte_frame[a:b+2]
            byte_frame = byte_frame[b+2:]

            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

                # Keep only last 5 seconds of frames
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(img)

# Run frame capture in a background thread


# FFmpeg command to capture stream
FFMPEG_CMD = [
    "ffmpeg", "-i", YOUTUBE_STREAM_URL, "-f", "image2pipe", "-vcodec", "mjpeg", "-"
]

# Frame buffer (5 seconds @ 30 FPS = 150 frames)
frame_queue = queue.Queue(maxsize=150)



import cv2
import numpy as np
import subprocess
import threading
import queue
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# YouTube Live Stream URL
YOUTUBE_STREAM_URL = "YOUTUBE_LIVE_STREAM_URL"

# FFmpeg command to capture stream
FFMPEG_CMD = [
    "ffmpeg", "-i", YOUTUBE_STREAM_URL, "-f", "image2pipe", "-vcodec", "mjpeg", "-"
]

# Frame buffer (5 seconds @ 30 FPS = 150 frames)
frame_queue = queue.Queue(maxsize=150)

# Load CLIP Model (pretrained)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Freeze all layers (fine-tuning only the last layers)
for param in clip_model.parameters():
    param.requires_grad = False

# Unfreeze last few layers for fine-tuning
for param in clip_model.visual.transformer.parameters():
    param.requires_grad = True  # Fine-tune only transformer layers

# Define optimizer & loss function
optimizer = optim.Adam(clip_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Function to capture frames
def capture_stream():
    process = subprocess.Popen(FFMPEG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    byte_frame = b""

    while True:
        byte_frame += process.stdout.read(1024)
        a = byte_frame.find(b'\xff\xd8')  # JPEG start
        b = byte_frame.find(b'\xff\xd9')  # JPEG end

        if a != -1 and b != -1:
            jpg = byte_frame[a:b+2]
            byte_frame = byte_frame[b+2:]

            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

                # Keep only last 5 seconds of frames
                if frame_queue.full():
                    frame_queue.get()
                frame_queue.put(img)

# Run frame capture in a background thread
threading.Thread(target=capture_stream, daemon=True).start()

# Function to fine-tune model & save checkpoints
def train_ai_on_frames():
    checkpoint_interval = 100  # Save model every 100 frames
    frame_count = 0

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Convert frame to tensor
            image = Image.fromarray(frame)
            image_tensor = transform(image).unsqueeze(0)

            # Forward pass through AI model
            optimizer.zero_grad()
            inputs = clip_processor(images=image, return_tensors="pt")
            outputs = clip_model(**inputs)

            # Dummy label (replace with real training labels)
            label = torch.tensor([1]).long()
            loss = criterion(outputs.logits_per_image, label)

            # Backpropagation & optimization
            loss.backward()
            optimizer.step()

            frame_count += 1
            print(f"Processed {frame_count} frames, Loss: {loss.item()}")

            # Save model every 100 frames
            if frame_count % checkpoint_interval == 0:
                torch.save(clip_model.state_dict(), f"clip_finetuned_{frame_count}.pt")
                print(f"Checkpoint saved: clip_finetuned_{frame_count}.pt")

# Run training in a background thread
threading.Thread(target=train_ai_on_frames, daemon=True).start()
