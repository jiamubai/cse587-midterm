import torch.nn as nn
class CNNModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, max_len=128):
        super(CNNModel, self).__init__()
        self.conv = nn.Conv1d(embedding_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # self.fc = nn.Linear(128 * (embedding_dim // 2), num_classes)
        self.fc = nn.Linear(128 * (max_len // 2), num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, n_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        x = self.fc(hidden[-1])
        return x