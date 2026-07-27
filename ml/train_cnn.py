import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import SolarFlareDataset

# 1. 1D Convolutional Neural Network Engine Block
# 1. 1D Convolutional Neural Network Engine Block
class FlareBaselineCNN(nn.Module):
    def __init__(self, window_size=60):
        super(FlareBaselineCNN, self).__init__()
        # Input shape expected: (Batch, Features, Sequence_Length) -> (B, 2, 60)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Output: (B, 16, 30)
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Structural reduction to static spatial dimension -> (B, 32, 1)
        )
        self.dense_out = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Resolve bounded probabilities [0.0 - 1.0]
        )
        
    def forward(self, x):
        # Permute input shapes matching PyTorch channel bounds: (B, 60, 2) -> (B, 2, 60)
        x = x.permute(0, 2, 1)
        features = self.conv_stack(x).squeeze(-1)
        return self.dense_out(features)


# 2. Execution and Optimization Routine
def run_training_pipeline():
    # Configure path targets matching your directory rules
    base_dir = r"C:\Users\DELL\OneDrive\Desktop\KARN\data"
    catalog_path = os.path.join(base_dir, "processed", "full_catalog.csv")
    flux_path = os.path.join(base_dir, "raw", "goes_flux_combined.csv")

    
    print("Initializing Data Ingestion Arrays...")
    dataset = SolarFlareDataset(catalog_path, flux_path)
    
    # Stratified Train/Val split boundaries
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    
    model = FlareBaselineCNN()
    
    # Skew pos_weight to penalize the loss function heavily if any true flare is missed
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0])) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Commencing Model Optimization Training Loop...")
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # Use raw uncalibrated tracking elements for stable backprop
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        print(f"Epoch {epoch}/10 | Target Pipeline Loss: {running_loss / len(train_loader.dataset):.4f}")
        
    print("\nTraining Complete! Saving Baseline Checkpoint...")
    torch.save(model.state_dict(), "baseline_cnn.pth")
    print("Checkpoint file baseline_cnn.pth successfully exported to root directory.")

if __name__ == "__main__":
    run_training_pipeline()
