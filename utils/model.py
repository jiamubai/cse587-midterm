import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, max_len=128, kernel_size=5):
        super(CNNModel, self).__init__()
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # self.fc = nn.Linear(hidden_dim * (embedding_dim // 2), num_classes)
        self.fc = nn.Linear(hidden_dim * (max_len // 2), num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define RNN model
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, n_layers=1):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        hidden = self.dropout(hidden[-1])
        x = self.fc(hidden)
        # if self.fc.out_features == 1:
        #     x = torch.sigmoid(x)  # Binary classification
        # else:
        #     x = F.log_softmax(x, dim=1)  # Multi-class classification
        return x


class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, n_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # RNN only returns (output, hidden), no cell state like LSTM
        _, hidden = self.rnn(x)
        hidden = self.dropout(hidden[-1])
        x = self.fc(hidden)
        return x
