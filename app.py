import json
import torch
import streamlit as st
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).parent

with open(BASE_DIR / "class_names.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

class_names = meta["class_names"]
img_size = meta["image_size"]
arch = meta["architecture"]

def build_classifier(in_f, num_c):
    return nn.Sequential(
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_c)
    )

@st.cache_resource
def load_model():
    device = torch.device("cpu")

    if arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = build_classifier(
            model.classifier[-1].in_features,
            len(class_names)
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    model.load_state_dict(
        torch.load(
            BASE_DIR / "fish_disease_mobilenet_v3_small.pt",
            map_location=device
        )
    )
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

st.title("Fish Disease Identification")

uploaded_file = st.file_uploader(
    "Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_idx = model(x).argmax(1).item()

    st.subheader("Prediction")
    st.write(class_names[pred_idx])