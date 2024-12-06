import torch
from torch.utils.data import Dataset
# Child stock dataset class inheriting from torch Dataset
# Need to override __len__ function and __getitem__
class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1 

    def __getitem__(self, i):
        prev_data = self.data[i: i + self.seq_len]
        future_data = self.data[i + self.seq_len: i + self.seq_len + 1]
        # Return the values as tensors
        return torch.tensor(prev_data, dtype=torch.float32), torch.tensor(future_data, dtype=torch.float32)
