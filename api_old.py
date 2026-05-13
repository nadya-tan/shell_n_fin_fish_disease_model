from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import json
from pathlib import Path
from pillow_heif import register_heif_opener

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shell-and-fin.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "class_names.json", "r") as f:
    meta = json.load(f)

class_names = meta["class_names"]
img_size = meta["image_size"]

def build_classifier(in_f, num_c):
    return nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_c)
    )

def load_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = build_classifier(
        model.classifier[-1].in_features,
        len(class_names)
    )
    model.load_state_dict(
        torch.load(BASE_DIR / "fish_disease_mobilenet_v3_small.pt", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    register_heif_opener()
    image = Image.open(file.file).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return {
        "prediction": class_names[pred_idx],
        "confidence": float(probs[pred_idx])
    }