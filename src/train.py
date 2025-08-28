# Import the Necessary Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from model import get_resnet50

# Setup the device for GPU usage if active
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Processed Data
train_data = torch.load("data/processed/train.pt")
val_data = torch.load("data/processed/val.pt")

# Wrap the dataset  
batch_size=32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


# Model Initialization
model = get_renset50(num_classes=10, pretrained=True, freeze_backbone=True)
model = model.to(device)


# Loss and Optimizer functions
getloss = nn.CrossEntropyLoss()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Accuracy Function
def calculate_accuracy(predictions, true_labels):


# Training Loop
def train_batch(x, y, model, optimizer, getloss):
    
    model.train()
    
    prediction = model(x)
    
    batch_loss = getloss(prediction, y)
    
    optimizer.zero_grad()
    
    batch_loss.backward()
    
    optimizer.step()
    
    return batch_loss.item()


#Validation Loop
def validate_batch(x, y, model, getloss):