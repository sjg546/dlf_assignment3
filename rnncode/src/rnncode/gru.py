import torch
import torch.nn as nn
# GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Initialise GRU model
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Add fully connected final layer
        self.fc = nn.Linear(hidden_size, output_size)

    # Define forward pass function
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out