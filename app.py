import torch
import requests
import os

MODEL_URL = "https://huggingface.co/Ameen2004/Catract-Eye-Disease-Detection/blob/main/hybrid_model.pth"

if not os.path.exists("hybrid_model.pth"):
    r = requests.get(MODEL_URL)
    with open("hybrid_model.pth", "wb") as f:
        f.write(r.content)

model.load_state_dict(torch.load("hybrid_model.pth", map_location=torch.device("cpu")))
model.eval()