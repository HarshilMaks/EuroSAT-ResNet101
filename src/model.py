import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

#Create a ResNet50 model for classification
def get_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(weights=None)
    
    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for regularization
        nn.Linear(512, num_classes)
    )
    
    # Freeze backbone if requested
    if freeze_backbone:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the final classifier layers
        for param in model.fc.parameters():
            param.requires_grad = True
        
        print("Backbone frozen. Only final classifier layers will be trained.")
    else:
        print("Full model will be trained (fine-tuning).")
        
    return model


def get_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params