import torch
import torch.nn as nn

class KarnMultiTaskEngine(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=64, num_classes=4):
        super(KarnMultiTaskEngine, self).__init__()
        
        # 1D-CNN Layer to capture immediate short-term feature changes
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels= hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # LSTM Layer to track long-term historical context
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        
        # Multi-Task Head A: Classification (Quiet, C, M, X)
        self.classifier_head = nn.Linear(hidden_dim, num_classes)
        
        # Multi-Task Head B: Continuous Time-to-Peak Regression
        self.regression_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # Input shape: [Batch, Seq_Len, Features] -> [B, 30, 23]
        # Transpose for Conv1d expectation: [Batch, Features, Seq_Len]
        x_conv = x.transpose(1, 2)
        x_conv = self.relu(self.conv1d(x_conv))
        
        # Transpose back for LSTM input loops
        x_lstm_in = x_conv.transpose(1, 2)
        
        # Extract hidden states across sequence positions
        lstm_out, (hn, cn) = self.lstm(x_lstm_in)
        
        # Isolate the final time-step hidden layer ('NOW')
        final_state = lstm_out[:, -1, :]
        
        # Compute forward pass output targets
        class_logits = self.classifier_head(final_state)
        time_to_peak = self.regression_head(final_state)
        
        return class_logits, time_to_peak

if __name__ == "__main__":
    # Structural verification check
    model = KarnMultiTaskEngine()
    test_tensor = torch.randn(2, 30, 23)
    logits, reg = model(test_tensor)
    print("Multi-Task Verification Output shapes:")
    print("-> Class Logits Tensor Shape:", logits.shape)
    print("-> Regression Time-to-Peak Shape:", reg.shape)
