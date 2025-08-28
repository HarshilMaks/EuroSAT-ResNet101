from datasets import Dataset
from model import get_resnet50
from preprocess import EuroSATDataset, GetDataLoaders

def main():
    print("Testing imports...")
    
    # Test your preprocessing
    processor = GetDataLoaders(Dataset)
    processor.main()  # whatever method you want to call
    
    # Test your model
    model = get_resnet50(num_classes=10, pretrained=True, freeze_backbone=True)
    print(f"THIS: {model} WORKSSSSSSS!!!!!")

if __name__ == "__main__":
    main()