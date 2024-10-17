import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



class StockDataset(Dataset):
    def __init__(self, X, y):
        """
        Initializes the dataset with inputs and targets.

        Parameters:
        - X (np.ndarray): Input sequences.
        - y (np.ndarray): Targets.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(datasets, batch_size=32, window_size=None, horizon=None, stock=None):
  if(window_size is not None and horizon is not None):

    key = (window_size, horizon)
    if stock is not None:
      key = (window_size, horizon, stock)

    dataloaders = {}
    dataloaders[key] = {}
    for split in ['train', 'val', 'test']:
      X = datasets[key][split]['x']
      y = datasets[key][split]['y']
      dataset = StockDataset(X, y)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
      dataloaders[key][split] = dataloader
    return dataloaders

  dataloaders = {}
  for key in datasets.keys():
      dataloaders[key] = {}
      for split in ['train', 'val', 'test']:
          X = datasets[key][split]['x']
          y = datasets[key][split]['y']
          dataset = StockDataset(X, y)
          dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
          dataloaders[key][split] = dataloader
  return dataloaders



def create_sequences(data, window_size, horizon, target_stock=None):
  """
  Creates time series input-output

  Parameters:
    - data : Normalized stock prices
    - window_size: Number of past days to consider
    - horizon: Number of days ahead to predict
  Returns
    - x: Input sequences
    - y: Target sequences
  """
  x, y = [], []
  if target_stock is None:
    total_samples = (len(data) - window_size - horizon + 1)
    for t in range(0, total_samples, window_size):
      x_t = data.iloc[t:t+window_size].values
      y_t = data.iloc[t + window_size + horizon  - 1].values
      x.append(x_t)
      y.append(y_t)
    x = np.array(x)
    y = np.array(y)
  else:
    total_samples = (len(data) - window_size - horizon + 1)
    for t in range(0, total_samples, window_size):
      x_t = data.iloc[t:t+window_size][target_stock].values
      x_t = x_t.reshape(-1, 1)
      y_t = data.iloc[t + window_size + horizon  - 1][target_stock]
      x.append(x_t)
      y.append(y_t)
    x = np.array(x)
    y = np.array(y).reshape(-1, 1)
  return x, y


def train_val_test_split(x, y, train_ratio=0.7, val_ratio=0.15):
  """
  Splits the data into train, validation, and test sets
  """
  total_samples = len(x)
  train_end     = int(total_samples * train_ratio)
  val_end       = train_end + int(total_samples * val_ratio)
  X_train       = x[:train_end]
  y_train       = y[:train_end]
  X_val         = x[train_end:val_end]
  y_val         = y[train_end:val_end]
  X_test        = x[val_end:]
  y_test        = y[val_end:]

  return X_train, y_train, X_val, y_val, X_test, y_test

def create_datasets(data, window_sizes, horizons, stocks=None):
  """
  Creates datasets for different window sizes and horizons
  """
  datasets = {}
  if stocks is None:
    for window_size in window_sizes:
      for horizon in horizons:
        x, y = create_sequences(data, window_size, horizon)
        datasets[(window_size, horizon)] = {'x': x, 'y': y}
        #print(f"Dataset for window size {window_size} and horizon {horizon} created.")
  else:
      for window_size in window_sizes:
        for horizon in horizons:
          for stock in stocks:
            x, y = create_sequences(data, window_size, horizon, stock)
            datasets[(window_size, horizon, stock)] = {'x': x, 'y': y}
            print(f"Dataset for stock {stock}, window size {window_size} and horizon {horizon} created.")


  for key in datasets.keys():
      X = datasets[key]['x']
      y = datasets[key]['y']
      X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
      datasets[key]['train'] = {'x': X_train, 'y': y_train}
      datasets[key]['val'] = {'x': X_val, 'y': y_val}
      datasets[key]['test'] = {'x': X_test, 'y': y_test}
      del datasets[key]['x']
      del datasets[key]['y']
      print(f" - Split into Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
  return datasets