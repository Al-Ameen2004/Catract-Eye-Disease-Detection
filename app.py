import torch
import requests
import os

MODEL_URL = "https://huggingface.co/Ameen2004/Catract-Eye-Disease-Detection/resolve/main/hybrid_model.pth"
MODEL_PATH = "hybrid_model.pth"

# Create model architecture
model = HybridModel()   # <-- use your real class name

# Download model if not present
if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

# Load weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()