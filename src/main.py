from datasets import Dataset
from model import get_resnet101
from preprocess import EuroSATDataset, GetDataLoaders
from torchvision.models import ResNet101_Weights

def main():
    print("Testing imports...")
    
    # Test your preprocessing
    processor = GetDataLoaders(Dataset)
    processor.main()
    
    # Test your model
    model = get_resnet101(
        num_classes=10,
        weights=ResNet101_Weights.DEFAULT,  # pretrained=True equivalent
        freeze_backbone=True
    )
    print(f"THIS: {model} WORKSSSSSSS!!!!!")

if __name__ == "__main__":
    main()
