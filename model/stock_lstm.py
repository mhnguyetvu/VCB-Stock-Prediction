import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm4 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x, _ = self.lstm4(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
