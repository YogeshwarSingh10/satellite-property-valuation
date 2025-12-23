import torch
import torch.nn as nn
from torchvision import models


def load_efficientnet(device: str):
    """
    Loads EfficientNet-B0 pretrained on ImageNet
    and returns a frozen feature extractor that outputs 1280-D embeddings.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Remove classifier (1000-class head)
    model.classifier = nn.Identity()

    model.eval()
    model.to(device)

    return model
