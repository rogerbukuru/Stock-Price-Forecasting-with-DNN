import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import time
from situation_analysis.prediction_model import GraphWaveNet
from monitor.data_processor import create_datasets
from monitor.data_processor import create_dataloaders
from monitor.data_processor import create_datasets_with_walk_forward
import copy
import time



class ContinualLearningPipeline:
    def __init__(self, model, val_loss, test_loss, window_size, horizon, buffer_size=1000, batch_size=32, lr=0.01, drift_threshold=0.2, max_training_runs=3, ):
        self.model = model
        self.updated_model = model
        self.updated_val_loss = val_loss
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
        self.current_val_loss = val_loss
        self.current_test_loss = test_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.L1Loss()  # Default criterion (MAE)
        self.epochs    = 100
        self.window_size = window_size
        self.horizon = horizon

    def add_to_buffer(self, new_data, walk_forward_step):
        """
        Add new data to the buffer and remove old data if the buffer exceeds buffer_size.
        """
       
        self.buffer.append(new_data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        
        scaler          = StandardScaler()
        buffer_data     = pd.concat(self.buffer, ignore_index=True)
        data_normalized = scaler.fit_transform(buffer_data)
        data_normalized = pd.DataFrame(data_normalized, columns=buffer_data.columns)


        stocks             = data_normalized.columns.tolist()
        window_sizes       = [30, 60, 120]
        horizons           = [1, 2, 5, 10, 30]
        # use walk-forward validation to create datasets
        datasets           = create_datasets_with_walk_forward(data=data_normalized, window_sizes=window_sizes, horizons=horizons)
        dataloaders        = create_dataloaders(datasets, batch_size = 32, window_size = self.window_size, horizon = self.horizon, step=walk_forward_step)
        selected_key       = (self.window_size, self.horizon, f'step_{walk_forward_step}')
       
        print(f"Walk forward step: {walk_forward_step}")
        self.train_loader = dataloaders[selected_key]['train']
        self.val_loader   = dataloaders[selected_key]['val']
        self.test_loader  = dataloaders[selected_key]['test']

    def warm_start_training(self):
        """
        Train a new model using the data from the buffer.
        """
        # Call GWN Training 
        print("Warm start training")
        loss_drift = float("-inf")
        for _ in range(self.max_training_runs):
            if self.val_loader is not None:
                print("Training")
                train_loss, new_val_loss = self.train(self.epochs)
                loss_drift =+ self.calculate_drift(new_val_loss, self.current_val_loss)
            avg_loss_drift =   loss_drift/self.max_training_runs
            print(f"Avg Validation Loss drift: {avg_loss_drift}")  
            if avg_loss_drift < self.drift_threshold:
                print("Within Drift Threshold")
                return None
            else: 
                print("Model Drift Detected")
                self.updated_val_loss = new_val_loss
                return self.updated_model
                

    def train(self, epochs):
        avg_train_loss = float('-inf')
        val_loss = float('-inf')
        total_train_time = 0 
        print("Start model training")
        for epoch in range(epochs):
            start_time = time.time()  # Start time for the epoch
            self.updated_model.train()
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
                    output = self.updated_model(x_batch)
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
                val_loss, avg_valid_mape, avg_valid_rmse = self.validate()
                # End of epoch log
                epoch_duration = time.time() - start_time
                total_train_time += epoch_duration
                print("--------------------------------------------------------------------------------")
                print(f"end of epoch {epoch + 1:<3} | time: {epoch_duration:>5.2f}s | "
                    f"train MAE {avg_train_loss:>6.4f} | valid MAE {val_loss:>6.4f}")
                print(f"train MAPE: {avg_train_mape:>6.4f} | valid MAPE: {avg_valid_mape:>6.4f}")
                print(f"train RMSE: {avg_train_rmse:>6.4f} | valid RMSE: {avg_valid_rmse:>6.4f}")
                print("--------------------------------------------------------------------------------")


        return avg_train_loss, val_loss

     
    def validate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.updated_model.eval()
        valid_loss = 0
        valid_mape = 0
        valid_rmse = 0
        valid_batches = 0
        if self.val_loader is not None:
            with torch.no_grad():
                for x, y in self.val_loader:
                    x_batch, y_batch = x.to(device), y.to(device)
                    output = self.updated_model(x_batch)
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
        return avg_valid_loss, avg_valid_mape, avg_valid_rmse

    def calculate_drift(self, new_loss, old_loss):
        """
        Calculate the drift in validation loss over time.
        """
        return abs(new_loss - old_loss)

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
    
    def continual_learning_step(self, new_data, walk_forward_step):
        """
        Execute a single continual learning step.
        """
        # Step 1: Add new data to buffer
        self.add_to_buffer(new_data, walk_forward_step)
        model_path = f'situation_analysis/updated_model/model_window30_horizon1.pth'
        
        # Step 2: Warm start training on updated buffer
        updated_model = self.warm_start_training()
        if updated_model is not None: # we have a new model
            # Step 3: Run model stability test
            test_loss = self.model_stability_test(updated_model)
            if self.calculate_drift(test_loss, self.current_test_loss) > self.drift_threshold:
                # Proceed to use the updated model if stability consistent
                print("Model stability is good. Saving new model")
                torch.save({
                    'model_state_dict': updated_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': self.updated_val_loss
                }, model_path)  
                return updated_model
            else:
                # Stability inconsistent; return error indicating that full model training requiredperform full model retraining
                print("Stability inconsistent returning existing model as full model retraining is required.")
                torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': self.current_val_loss,
                'test_loss': self.current_test_loss
                }, model_path)   
                raise Exception("New moel stability is inconsistent with previous model, full model retraining may be required.")
        
        else:
                # we keep our existing model 
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': self.current_val_loss,
                'test_loss': self.current_test_loss
            }, model_path)      


    def mean_absolute_error(self,y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred)).item()

    def mean_absolute_percentage_error(self,y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)).item() * 100

    def root_mean_square_error(self,y_pred, y_true):
        return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()



def load_model(model_path, device):
    """
    Load a model from a .pth file.
    """
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    config = {
            'model_path': model_path,
            'num_nodes': state_dict['adapt_adj'].shape[0],
            'hidden_size': state_dict['gc1.fc.weight'].shape[0],
            'dropout_rate': state_dict.get('dropout.p', checkpoint.get('dropout_rate', 0.1)),
            'window_size': checkpoint.get('window_size', 
                int(model_path.split('window')[1].split('_')[0])),
            'prediction_horizon': checkpoint.get('prediction_horizon',
                int(model_path.split('horizon')[1].split('.')[0])),
            'learning_rate': checkpoint.get('learning_rate', 0.01)
        }
    
    print(f"Keys:{config.keys()}")
    best_val_loss          = 0.0368
    test_loss              = 0.0354
    input_size             = config['window_size']
    prediction_horizon     = config['prediction_horizon']
    hidden_size            = config['hidden_size']
    learning_rate          = config['learning_rate']
    dropout_rate           = config['dropout_rate']
    num_nodes              = config['num_nodes']
    print(f"Best val loss: {best_val_loss} ")

    
    # Instantiate the model with the saved architecture
    model = GraphWaveNet(input_size=36, hidden_size=hidden_size, output_size=36, num_nodes=num_nodes, dropout_rate=dropout_rate)
    
    # Load the model state (weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    
    print(f"Model successfully loaded from {model_path}")
    
    return model, best_val_loss, test_loss

