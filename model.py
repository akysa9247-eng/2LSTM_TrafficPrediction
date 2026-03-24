# model.py
import torch
import torch.nn as nn

class TrafficLSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(TrafficLSTM_model, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln_fc = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out, _ = self.lstm2(out)
        out = self.ln2(out)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.ln_fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out