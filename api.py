from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import json
from pathlib import Path
from pillow_heif import register_heif_opener

# Initialize FastAPI application
app = FastAPI()

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shell-and-fin.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the base directory of the current file
BASE_DIR = Path(__file__).parent

# Load class names and image size from config file
with open(BASE_DIR / "class_names.json", "r") as f:
    meta = json.load(f)

class_names = meta["class_names"]
img_size = meta["image_size"]

# Build custom classifier head for the model
def build_classifier(in_f, num_c):
    return nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_c)
    )

# Load pre-trained model and set to evaluation mode
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    # Replace the final classification layer
    model.classifier[-1] = build_classifier(
        model.classifier[-1].in_features,
        len(class_names)
    )
    # Load trained weights to CPU
    model.load_state_dict(
        torch.load(BASE_DIR / "fish_disease_mobilenet_v3_small.pt", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# API endpoint for image prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Enable HEIF image format support
    register_heif_opener()
    # Open and convert image to RGB
    image = Image.open(file.file).convert("RGB")
    # Apply transforms and add batch dimension
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

        # Get top-2 class indices and confidence scores in descending order
        top_probs, top_indices = torch.topk(probs, k=2)
        top1_idx = top_indices[0].item()
        top1_prob = top_probs[0].item()
        top2_idx = top_indices[1].item()
        top2_prob = top_probs[1].item()

    # Get class name for top-1 prediction
    top1_name = class_names[top1_idx]

    if top1_name == "Healthy":
        # Rule 1: Top prediction is Healthy: return only Healthy
        return {
            "status": "healthy",
            "result": [{"disease": top1_name, "confidence": round(top1_prob, 4)}]
        }
    else:
        if top1_prob >= 0.7:
            # Rule 2: Disease confidence >= 70%: return only the top disease
            return {
                "status": "single_disease",
                "result": [{"disease": top1_name, "confidence": round(top1_prob, 4)}]
            }
        else:
            # Rule 3: Disease confidence <70%: return top 2 possible diseases
            top2_name = class_names[top2_idx]
            return {
                "status": "possible_multiple",
                "result": [
                    {"disease": top1_name, "confidence": round(top1_prob, 4)},
                    {"disease": top2_name, "confidence": round(top2_prob, 4)}]
            }