import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import os
import itertools

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return data, normalized_data, scaler

def create_sequences(data, seq_length, horizon):
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        x = data[i:i + seq_length]
        y = data[i + seq_length + horizon - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.fc(out)
        return out

class GraphWaveNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_nodes, dropout_rate=0.2):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.adapt_adj = nn.Parameter(torch.eye(num_nodes))
        self.gc1 = GraphConvLayer(input_size, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.temporal_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        adj = self.adapt_adj
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gc2(x, adj)
        x = torch.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        out = self.fc(x[:, -1, :])
        return out

class ContinualLearningModel:
    def __init__(self, input_size, hidden_size, output_size, num_nodes, learning_rate, dropout_rate):
        self.model = GraphWaveNet(input_size, hidden_size, output_size, num_nodes, dropout_rate).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.best_val_loss = float('inf')

    def update_model(self, train_data, train_labels, val_data, val_labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(train_data)
        loss_fn = nn.MSELoss()
        loss = loss_fn(outputs.squeeze(), train_labels)
        loss.backward()
        self.optimizer.step()
        val_loss = self.validate(val_data, val_labels)
        return loss.item(), val_loss

    def validate(self, val_data, val_labels):
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(val_data)
            val_loss_fn = nn.MSELoss()
            val_loss = val_loss_fn(val_outputs.squeeze(), val_labels)
        return val_loss.item()

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)

def run_continuous_learning_component(data_path, input_window, prediction_horizon, hidden_size, learning_rate, dropout_rate, epochs):
    data, normalized_data, scaler = load_and_preprocess_data(data_path)
    
    X_seq, y_seq = create_sequences(normalized_data, input_window, prediction_horizon)
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    
    model = ContinualLearningModel(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        output_size=y_train.shape[1],
        num_nodes=X_train.shape[1],
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )
    
    best_val_loss = float('inf')
    for epoch in tqdm(range(epochs), desc=f"Training (Window: {input_window}, Horizon: {prediction_horizon}, Hidden: {hidden_size}, LR: {learning_rate}, Dropout: {dropout_rate})"):
        train_loss, val_loss = model.update_model(X_train, y_train, X_val, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return best_val_loss, model

def train_multiple_models(data_path, input_windows, prediction_horizons, hidden_sizes, learning_rates, dropout_rates, epochs):
    # Create the best_models directory if it doesn't exist
    os.makedirs('best_models', exist_ok=True)
    
    best_models = {}
    
    for input_window, prediction_horizon in itertools.product(input_windows, prediction_horizons):
        print(f"\nTuning hyperparameters for Input Window: {input_window}, Prediction Horizon: {prediction_horizon}")
        
        best_val_loss = float('inf')
        best_params = None
        best_model = None
        
        for hidden_size, learning_rate, dropout_rate in itertools.product(hidden_sizes, learning_rates, dropout_rates):
            val_loss, model = run_continuous_learning_component(
                data_path=data_path,
                input_window=input_window,
                prediction_horizon=prediction_horizon,
                hidden_size=hidden_size,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                epochs=epochs
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = (hidden_size, learning_rate, dropout_rate)
                best_model = model
        
        print(f"Best parameters for Window: {input_window}, Horizon: {prediction_horizon}")
        print(f"Hidden Size: {best_params[0]}, Learning Rate: {best_params[1]}, Dropout Rate: {best_params[2]}")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        
        model_path = f'best_models/model_window{input_window}_horizon{prediction_horizon}.pth'
        best_model.save_model(model_path)
        print(f"Best model saved to {model_path}")
        
        best_models[(input_window, prediction_horizon)] = {
            'params': best_params,
            'val_loss': best_val_loss,
            'model_path': model_path
        }
    
    return best_models

if __name__ == "__main__":
    best_models = train_multiple_models(
        data_path='data/JSE_clean_truncated.csv',
        input_windows=[30, 60, 120],
        prediction_horizons=[1, 2, 5, 10, 30],
        hidden_sizes=[32, 64, 128],
        learning_rates=[0.001, 0.01, 0.1],
        dropout_rates=[0.1, 0.2, 0.3],
        epochs=50  # Reduced epochs for faster hyperparameter tuning
    )
    
    # Print summary of best models
    print("\nSummary of Best Models:")
    for (input_window, prediction_horizon), model_info in best_models.items():
        print(f"Window: {input_window}, Horizon: {prediction_horizon}")
        print(f"Best Parameters: Hidden Size: {model_info['params'][0]}, Learning Rate: {model_info['params'][1]}, Dropout Rate: {model_info['params'][2]}")
        print(f"Best Validation Loss: {model_info['val_loss']:.4f}")
        print(f"Model saved at: {model_info['model_path']}")
        print()