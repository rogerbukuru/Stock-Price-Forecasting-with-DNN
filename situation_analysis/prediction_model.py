import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Graph Convolutional Layer
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, features = x.size()
        x = x.view(batch_size * seq_len * num_nodes, features)
        out = self.fc(x)
        out = out.view(batch_size, seq_len, num_nodes, -1)
        return out

# Graph Wavenet model
class GraphWaveNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_nodes, dropout_rate=0.2):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.gc1 = GraphConvLayer(input_size, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.temporal_conv = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(1, 3), padding=(0, 1))
        self.fc = nn.Linear(hidden_size * num_nodes, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_len, num_nodes = x.size()
        x = x.unsqueeze(-1)  # Add feature dimension
        
        x = self.gc1(x, None)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, None)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = x.permute(0, 3, 1, 2)  # [batch_size, hidden_size, seq_len, num_nodes]
        x = self.temporal_conv(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 3, 1)  # [batch_size, seq_len, num_nodes, hidden_size]
        
        x = x[:, -1, :, :]  # Use only the last time step
        x = x.reshape(batch_size, -1)  # Flatten
        out = self.fc(x)
        return out