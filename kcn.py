# kcn.py
import torch
import torch.nn as nn
from torchvision import models

from CNN_KAN.kan_layer import KANLinear  # absolute import via package

class ConvNeXtKAN(nn.Module):
    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True):
        super().__init__()

        # gunakan pretrained weights yang benar
        try:
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            self.convnext = models.convnext_tiny(weights=weights)
        except Exception:
            # fallback untuk versi torchvision lama
            self.convnext = models.convnext_tiny(pretrained=True)

        if freeze_backbone:
            for p in self.convnext.parameters():
                p.requires_grad = False

        # Ambil dimensi fitur dari fully-connected terakhir dan matikan FC-nya
        # classifier = [LayerNorm2d(768, ...), Flatten(1), Linear(768, 1000)]
        num_features = self.convnext.classifier[-1].in_features
        self.convnext.classifier[-1] = nn.Identity()   # buang Linear(…, 1000)
        # Sampai di sini, forward convnext akan mengembalikan vektor 768 (sudah flatten)

        # Head KAN
        self.kan1 = KANLinear(num_features, 256)
        self.kan2 = KANLinear(256, num_classes)

    def forward(self, x):
        x = self.convnext(x)   # shape [B, 768]
        x = self.kan1(x)
        x = self.kan2(x)
        return x

def print_parameter_details(model: nn.Module):
    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            total += p.numel()
            print(f"{n}: {p.numel()}")
    print(f"Total trainable parameters: {total}")
