import torch
from models.faceboxes import FaceBoxes

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Create model
model = FaceBoxes(phase='train', size=1024, num_classes=2)

total, trainable = count_parameters(model)

print("="*60)
print("Parame Count: Basic with CBAM")
print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print("="*60)