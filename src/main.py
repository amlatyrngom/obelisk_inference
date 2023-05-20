import sys
import os

print(f"CWD: {os.getcwd()}")

if os.path.exists("venv"):
    curr_dir = os.getcwd()
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    sys.path.append(f"{curr_dir}/venv/lib/python{version}/site-packages")
    print(f"New Path: {sys.path}")

import torch
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import ToTensor

torch.hub.set_dir("/tmp/torch")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
preprocess = weights.transforms(antialias=True)
model = mobilenet_v3_small(weights=weights)
to_tensor = ToTensor()
model.eval()

def main(*args):
    # Read input
    batch = []
    args = args[0]
    for (img, width, height) in args:
        img = np.frombuffer(img, dtype=np.uint8) # TODO: Find a way to optimize this part.
        img.resize((height, width, 3))
        img = to_tensor(img.copy())
        batch.append(preprocess(img))    
    batch = torch.stack(batch)
    # Predict.
    predictions = model(batch)
    # Return best candidates.
    class_ids = predictions.argmax(dim=1).tolist()
    category_names = [weights.meta["categories"][class_id] for class_id in class_ids]
    return category_names
