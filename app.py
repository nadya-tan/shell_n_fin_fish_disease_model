from flask import Flask, request
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import threading
import time
import os

app = Flask(__name__)


# Auto shutdown configuration to limit service running time
# Total running time: 7200 seconds = 1 hours
AUTO_STOP_SECONDS = 7200

def auto_stop_after_delay():
    '''
    Background thread function to automatically stop the service after a fixed delay.
    
    This function runs in a separate daemon thread, counts down the specified time,
    and forcibly exits the program once time is up.
    
    input:
      None (uses global constant AUTO_STOP_SECONDS)
      
    output:
      None (terminates the process on completion)
    '''
    # Print service lifetime reminder
    print(f"⏱ Service will automatically shut down after {AUTO_STOP_SECONDS//60} minutes.")
    # Wait for the specified duration
    time.sleep(AUTO_STOP_SECONDS)
    # Print shutdown notification
    print("\n 2 hours elapsed, API shutting down automatically.")
    # Force exit the program to stop the Flask server
    os._exit(0)

# Start a background daemon thread for auto shutdown
# Daemon=True ensures thread exits when main program exits
threading.Thread(target=auto_stop_after_delay, daemon=True).start()
# ====================================================================

# Load model configuration from JSON file
# Reads class names, input image size, and model architecture type
with open("class_names.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# Extract configuration parameters
class_names = meta["class_names"]    # List of fish disease classification labels
img_size = meta["image_size"]        # Model input image resolution (width = height)
arch = meta["architecture"]          # Backbone model architecture name

# Build custom classifier head for transfer learning
def build_classifier(in_f, num_c):
    '''
    Construct a custom fully connected classifier for fish disease classification.
    
    Replaces the original classifier of the pre-trained model with a custom network.
    
    input:
      in_f  - Number of input features from the backbone model
      num_c - Number of output classes (disease categories)
      
    output:
      nn.Sequential - Custom classifier module
    '''
    return nn.Sequential(
        nn.Linear(in_f, 256),        # First linear layer: input -> 256 hidden units
        nn.ReLU(),                  # ReLU activation function
        nn.Dropout(0.4),            # Dropout layer with 40% rate to prevent overfitting
        nn.Linear(256, num_c)       # Final linear layer: 256 hidden units -> output classes
    )

# Build complete fish disease detection model
def build_model(architecture, num_cls):
    '''
    Build the full detection model using specified backbone architecture.
    
    Currently supports only mobilenet_v3_small with custom classifier.
    
    input:
      architecture - Model backbone name
      num_cls      - Number of classification classes
      
    output:
      model - Complete PyTorch model with custom classifier
    '''
    if architecture == "mobilenet_v3_small":
        # Initialize pre-trained MobileNetV3 model without pre-trained weights
        model = models.mobilenet_v3_small(weights=None)
        # Replace final classifier with custom network
        model.classifier[-1] = build_classifier(model.classifier[-1].in_features, num_cls)
        return model

# Set computation device to CPU (compatible with Codespaces environment)
device = torch.device("cpu")
# Build model and move it to the computation device
model = build_model(arch, len(class_names)).to(device)
# Load trained model weights from local file
model.load_state_dict(torch.load("fish_disease_mobilenet_v3_small.pt", map_location=device))
# Set model to evaluation mode (disable dropout/batch norm training behavior)
model.eval()

# Define image preprocessing pipeline (matches training pipeline)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),    # Resize image to model input size
    transforms.ToTensor(),                      # Convert PIL image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], # Normalize with ImageNet mean
                         [0.229, 0.224, 0.225]) # Normalize with ImageNet std
])

# API endpoint for fish disease prediction
@app.route("/predict", methods=["POST"])
def predict():
    '''
    API endpoint to predict fish disease from uploaded image.
    
    Accepts image file via POST request, performs preprocessing and inference,
    returns the predicted disease category as plain text.
    
    input:
      image - Uploaded image file from request.files
      
    output:
      class name - Predicted fish disease label (string)
    '''
    # Get uploaded image file from request
    file = request.files["image"]
    # Open image and convert to standard RGB format
    img = Image.open(file.stream).convert("RGB")
    # Apply preprocessing transformations and add batch dimension
    x = transform(img).unsqueeze(0).to(device)
    
    # Disable gradient computation for inference (speed + memory saving)
    with torch.no_grad():
        # Forward pass and get predicted class index
        pred_idx = model(x).argmax(1).item()
    
    # Return corresponding class label
    return class_names[pred_idx]

# Root endpoint for service health check
@app.route("/")
def index():
    '''
    Health check endpoint to verify API is running properly.
    
    output:
      Status message indicating API is active and will auto-stop after 2 hours
    '''
    return "Fish API is running! Auto shutdown after 2 hours."

# Main program entry
if __name__ == "__main__":
    # Print startup information
    print("API started successfully!")
    print("Runtime limit: Auto shutdown after 2 hours")
    print("Manual stop: Press Ctrl + C\n")
    # Run Flask server on all network interfaces, port 5000
    app.run(host="0.0.0.0", port=5000)