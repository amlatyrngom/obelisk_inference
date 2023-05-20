# Download model.
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms import ToTensor


torch.hub.set_dir("/tmp/torch")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
preprocess = weights.transforms()
model = mobilenet_v3_small(weights=weights)
to_tensor = ToTensor()
model.eval()
