"""ResNet-101 implementation from scratch for EuroSAT classification."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BasicConv2d(nn.Module):
    """Basic convolution block with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-101."""
    expansion = 4
    
    def __init__(self, in_channels: int, channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = BasicConv2d(in_channels, channels, kernel_size=1)
        self.conv2 = BasicConv2d(channels, channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = BasicConv2d(channels, channels * self.expansion, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet101(nn.Module):
    """ResNet-101 implementation."""
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.2, hidden_size: int = 512):
        super(ResNet101, self).__init__()
        
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be between 0.0 and 1.0")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        
        self.in_channels = 64
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 23, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * Bottleneck.expansion, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        self._initialize_weights()
        
    def _make_layer(self, channels: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_channels != channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * Bottleneck.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * Bottleneck.expansion),
            )
        
        layers = []
        layers.append(Bottleneck(self.in_channels, channels, stride, downsample))
        self.in_channels = channels * Bottleneck.expansion
        
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_channels, channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


def get_resnet101(num_classes: int, dropout_rate: float = 0.2, 
                  hidden_size: int = 512, freeze_backbone: bool = False) -> ResNet101:
    """Create ResNet-101 model for classification."""
    
    model = ResNet101(num_classes, dropout_rate, hidden_size)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False
        print("Backbone frozen. Only classifier will be trained.")
    else:
        print("Full ResNet-101 model will be trained.")
    
    return model


def get_model_info(model: nn.Module) -> Tuple[int, int]:
    """Get model parameter information."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return total_params, trainable_params


if __name__ == "__main__":
    # Test the implementation
    model = get_resnet101(num_classes=10)
    
    # Test with dummy data
    dummy_input = torch.randn(2, 3, 224, 224)
    model.eval()
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    
    get_model_info(model)
    print("ResNet-101 ready for training.")