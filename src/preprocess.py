from datasets import load_from_disk
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch
import os

class EuroSATDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # already PIL image from EuroSAT
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

class GetDataLoaders(DataLoader):
    @staticmethod
    def main():
        # Load raw dataset
        dataset = load_from_disk("data/raw/eurosat_rgb")

        # Define preprocessing transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet50 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Standard ResNet normalization
        ])

        # Create PyTorch Dataset
        euro_dataset = EuroSATDataset(dataset["train"], transform=transform)

        # Split into train & val
        train_size = int(0.8 * len(euro_dataset))
        val_size = len(euro_dataset) - train_size
        train_dataset, val_dataset = random_split(euro_dataset, [train_size, val_size])

        # Materialize datasets into batched tensors and save
        os.makedirs("data/processed", exist_ok=True)

        def to_tensor_dict(ds, batch_size=256):
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
            imgs, labs = [], []
            for x, y in loader:
                imgs.append(x)
                labs.append(y)
            images = torch.cat(imgs, dim=0)
            labels = torch.cat(labs, dim=0).long()
            return {"images": images, "labels": labels}

        train_tensors = to_tensor_dict(train_dataset)
        val_tensors = to_tensor_dict(val_dataset)

        torch.save(train_tensors, "data/processed/train.pt")
        torch.save(val_tensors, "data/processed/val.pt")

        print("Preprocessing complete. Datasets saved in data/processed/")

if __name__ == "__main__":
    GetDataLoaders.main()