import torch
import torch.nn as nn
# RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # build torch RNN model, RELU chosen as activation function
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,nonlinearity='relu')
        # Add fully connected final layer
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # Define hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1:, :])
        return out