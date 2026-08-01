import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import SolarFlareDataset

# 1. True 4-Class Deep LSTM Sequence Engine
class FlareRecurrentLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super(FlareRecurrentLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2       
        )
        self.linear_interpretation = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4) # Out branches mapping exactly to: [0: Quiet, 1: C-Class, 2: M-Class, 3: X-Class]
        )
        
    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        final_time_step = lstm_out[:, -1, :]
        return self.linear_interpretation(final_time_step)

# 2. Asymmetric Risk Score Implementation Logic
def calculate_consequence_severity(probabilities):
    """
    Computes custom operational risk metrics using your master plan equation:
    Score = 1*P(C) + 10*P(M) + 100*P(X)
    """
    # Exclude index 0 (Quiet class) to isolate risk probabilities
    p_c = probabilities[1]
    p_m = probabilities[2]
    p_x = probabilities[3]
    
    severity_score = (1 * p_c) + (10 * p_m) + (100 * p_x)
    
    # Asymmetric Decision Matrix Alerts
    if p_x >= 0.3 or severity_score >= 30.0:
        alert_status = "CRITICAL RED ALERT: PROBABLE X-CLASS EVENT IMMINENT"
    elif p_m >= 0.5 or severity_score >= 5.0:
        alert_status = "AMBER WARNING: EXPECTED SIGNIFICANT M-CLASS FLARE"
    else:
        alert_status = "GREEN STATE: NORMAL WEATHER PROFILE"
        
    return severity_score, alert_status, p_c, p_m, p_x

# 3. Optimization and Training Routine
def run_recurrent_pipeline():
    base_dir = r"C:\Users\DELL\OneDrive\Desktop\KARN\data"
    catalog_path = os.path.join(base_dir, "processed", "full_catalog.csv")
    flux_path = os.path.join(base_dir, "raw", "goes_flux_combined.csv")
    
    print("Loading Multi-Channel Window Tensors with Real 4-Class Labels...")
    dataset = SolarFlareDataset(catalog_path, flux_path)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    
    model = FlareRecurrentLSTM()
    
    # CrossEntropyLoss automatically evaluates true integer class allocations (0, 1, 2, or 3)
    # Skewing weights slightly to handle the historical class scarcity
    class_weights = torch.tensor([1.0, 2.0, 5.0, 10.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Commencing Deep LSTM Memory Gradient Descent...")
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # labels tensor contains true values (0-3) directly from dataset.py
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        print(f"Epoch {epoch}/10 | Recurrent Network Loss: {running_loss / len(train_loader.dataset):.4f}")
        
    print("\nTraining Complete! Saving Recurrent Checkpoint...")
    torch.save(model.state_dict(), "karn_lstm_core.pth")
    
    # Showcase active tracking on a live sample window from validation data
    print("\nTesting Live Probability Tracking Logic on Sample Frame:")
    model.eval()
    with torch.no_grad():
        sample_input, sample_label = dataset[0]
        sample_input = sample_input.unsqueeze(0) # Form batch dimension (1, 60, 2)
        raw_logits = model(sample_input)
        
        # Apply Softmax to map logits into smooth distributions bounded to 1.0
        probabilities = torch.softmax(raw_logits, dim=-1).squeeze().tolist()
        
        score, alert, pc, pm, px = calculate_consequence_severity(probabilities)
        print(f"-> Probabilities: [Quiet: {probabilities[0]:.2%}, C: {pc:.2%}, M: {pm:.2%}, X: {px:.2%}]")
        print(f"-> Calculated Risk Severity Score: {score:.2f}")
        print(f"-> System Status Output: {alert}")

if __name__ == "__main__":
    run_recurrent_pipeline()
