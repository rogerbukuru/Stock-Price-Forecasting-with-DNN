import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from walk_forward_validation import create_datasets
from walk_forward_validation import create_dataloaders
from tqdm import tqdm
import time
from prediction_model import GraphWaveNet


class ContinualLearningPipeline:
    def __init__(self, model, buffer_size=1000, batch_size=32, lr=0.001, drift_threshold=0.05, max_training_runs=3):
        self.model = model
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.drift_threshold = drift_threshold
        self.max_training_runs = max_training_runs
        self.buffer = []
        self.validation_loss_history = []
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss()  # Default criterion (MAE)
        self.epochs    = 100

        
    def add_to_buffer(self, new_data):
        """
        Add new data to the buffer and remove old data if the buffer exceeds buffer_size.
        """
        self.buffer.append(new_data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(self.buffer) # TODO: Get data from buffer
        data_normalized = pd.DataFrame(data_normalized, columns=data_normalized.columns)


        stocks             = data_normalized.columns.tolist()
        window_sizes       = [30, 60, 120]
        horizons           = [1, 2, 5, 10, 30]
        datasets           = create_datasets(data = data_normalized, window_sizes=window_sizes, horizons=horizons)
        dataloaders        = create_dataloaders(datasets, batch_size = 32, window_size = 30, horizon = 10)
        selected_key       = (30, 10)
       
        
        self.train_loader = dataloaders[selected_key]['train']
        self.val_loader   = dataloaders[selected_key]['val']
        self.test_loader  = dataloaders[selected_key]['test']

    def warm_start_training(self):
        """
        Train a new model using the data from the buffer.
        """
        # Call GWN Training 
        for _ in range(self.max_training_runs):
            if self.val_loader is not None:
                train_loss, val_loss = self.train(self.epochs)
                self.validation_loss_history.append(val_loss / len(self.val_loader))
                # Stop training if validation loss is consistent
                if len(self.validation_loss_history) > 1:
                    loss_drift = self.calculate_drift()
                    if loss_drift < self.drift_threshold:
                        break
        return self.model

    def train(self, epochs):
        avg_train_loss = float('-inf')
        val_loss = float('-inf')
        for epoch in range(epochs):
            start_time = time.time()  # Start time for the epoch
            model.train()
            train_loss = 0

            # Additional metrics
            train_mape = 0
            train_rmse = 0
            train_batches = 0

            # Training phase
            if self.train_loader is not None:
                for batch_idx, (x, y) in enumerate(self.train_loader):
                    x_batch, y_batch = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    output = model(x_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    # Calculate additional metrics
                    train_mape += self.mean_absolute_percentage_error(output, y_batch)
                    train_rmse += self.root_mean_square_error(output, y_batch)
                    train_batches += 1

                # Average training metrics
                avg_train_loss = train_loss / len(self.train_loader)
                avg_train_mape = train_mape / train_batches
                avg_train_rmse = train_rmse / train_batches
                val_loss = self.validate()
        return avg_train_loss, val_loss

     
    def validate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        valid_loss = 0
        valid_mape = 0
        valid_rmse = 0
        valid_batches = 0
        if self.val_loader is not None:
            with torch.no_grad():
                for x, y in self.val_loader:
                    x_batch, y_batch = x.to(device), y.to(device)
                    output = model(x_batch)
                    loss   = self.criterion(output, y_batch)
                    valid_loss += loss.item()
                    # Calculate additional metrics
                    valid_mape += self.mean_absolute_percentage_error(output, y_batch)
                    valid_rmse += self.root_mean_square_error(output, y_batch)
                    valid_batches += 1

            # Average validation metrics
            avg_valid_loss = valid_loss / len(self.val_loader)
            avg_valid_mape = valid_mape / valid_batches
            avg_valid_rmse = valid_rmse / valid_batches
        return avg_valid_loss

    def calculate_drift(self):
        """
        Calculate the drift in validation loss over time.
        """
        if len(self.validation_loss_history) < 2:
            return 0.0
        return abs(self.validation_loss_history[-1] - self.validation_loss_history[-2]) / self.validation_loss_history[-2]

    def model_stability_test(self, model):
        """
        Run a stability test on the updated model and return the average test loss across 3 runs.
        """        
        test_losses = []
        for _ in range(3):
            test_loss = 0.0
            if self.test_loader is not None:
                with torch.no_grad():
                    for x, y in self.test_loader:
                        outputs = model(x)
                        loss = self.criterion(outputs, y)
                        test_loss += loss.item()
                test_losses.append(test_loss / len(self.test_loader))
        
        return sum(test_losses) / len(test_losses)
    
    def continual_learning_step(self, new_data):
        """
        Execute a single continual learning step.
        """
        # Step 1: Add new data to buffer
        self.add_to_buffer(new_data)
        
        # Step 2: Warm start training on updated buffer
        updated_model = self.warm_start_training()
        
        # Step 3: Run model stability test
        test_loss = self.model_stability_test(updated_model)
        if self.calculate_drift() > self.drift_threshold:
            # Proceed to use the updated model if stability consistent
            return updated_model
        else:
            # Stability inconsistent; perform full model retraining
            print("Stability inconsistent, perform full model retraining")
            return self.full_model_retraining()



    def full_model_retraining(self):
        """
        Implement full model retraining procedure.
        """
        # Here you can implement a full model retraining using all available data.
        print("Full model retraining to be implemented.")
        return self.model

    def mean_absolute_error(self,y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred)).item()

    def mean_absolute_percentage_error(self,y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100

    def root_mean_square_error(self,y_pred, y_true):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


model = GraphWaveNet(input_size =  1, hidden_size = 64 , output_size = 30, num_nodes = 30, dropout_rate=0.2)
pipeline = ContinualLearningPipeline(model)

# Example new data
data = pd.read_csv("JSE_clean_truncated.csv")
data.shape # 3146 daily closing prices for 30 stocks
pipeline.continual_learning_step(data)
