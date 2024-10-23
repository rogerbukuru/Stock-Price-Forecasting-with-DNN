from invest.preprocessing.simulation import simulate
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
from monitor.walk_forward_validation import create_datasets
from monitor.walk_forward_validation import create_dataloaders
import copy
import time


class ContinualLearningPipeline:
    def __init__(self, model, val_loss, test_loss, buffer_size=1000, batch_size=32, lr=0.01, drift_threshold=0.2, max_training_runs=3):
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
        self.epochs = 100
        self.simulate_noise = False

    def add_to_buffer(self, new_data):
        """
        Add new data to the buffer and remove old data if the buffer exceeds buffer_size.
        """
       
        self.buffer.append(new_data)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        
        scaler          = StandardScaler()
        buffer_data     = pd.concat(self.buffer, ignore_index=True)
        data_normalized = scaler.fit_transform(buffer_data) # TODO: Get data from buffer
        data_normalized = pd.DataFrame(data_normalized, columns=buffer_data.columns)

        stocks             = data_normalized.columns.tolist()
        window_sizes       = [30, 60, 120]
        horizons           = [1, 2, 5, 10, 30]
        datasets           = create_datasets(data = data_normalized, window_sizes=window_sizes, horizons=horizons)
        dataloaders        = create_dataloaders(datasets, batch_size = 32, window_size = 60, horizon = 10)
        selected_key       = (60, 10)
       
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
                #self.validation_loss_history.append(val_loss / len(self.val_loader))
                # Stop training if validation loss is consistent
                #if len(self.validation_loss_history) > 1:
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
        #if len(self.validation_loss_history) < 2:
        #    return float('-inf')
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
    
    def continual_learning_step(self, new_data):
        """
        Execute a single continual learning step.
        """
        # Step 1: Optionally apply noise to the new data
        if self.simulate_noise:
            print("Applying noise to the new data")
            new_data = simulate(new_data, frac=0.3, scale=1, method='std')
        
        # Step 2: Add the (possibly noisy) data to buffer
        self.add_to_buffer(new_data)
        model_path = f'situation_analysis/updated_model/model_window60_horizon10.pth'
        
        # Step 3: Warm start training on updated buffer
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
                # Stability inconsistent; perform full model retraining
                print("Stability inconsistent returning existing model as full model retraining is required.")
                torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': self.current_val_loss,
                'test_loss': self.current_test_loss
                }, model_path)   
                #return self.full_model_retraining()
        else:
                # we keep our existing model 
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': self.current_val_loss,
                'test_loss': self.current_test_loss
            }, model_path)      


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



# Function to load the model from the .pth file
def load_model(model_file_path, device):
    """
    Load a model from a .pth file.
    """
    checkpoint = torch.load(model_file_path, map_location=device)
    
    # Extract the model architecture parameters from the checkpoint
    optimizer_state_dict   = checkpoint['optimizer_state_dict']
    best_val_loss          = 0.0393 # checkpoint['best_val_loss'] # never gets updated
    test_loss              = 0.0413
    print(f"Best val loss: {best_val_loss} ")

    
    # Instantiate the model with the saved architecture
    model = GraphWaveNet(input_size=30, hidden_size=128, output_size=30, num_nodes=60, dropout_rate=0.2)
    
    # Load the model state (weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    
    print(f"Model successfully loaded from {model_file_path}")
    
    return model, best_val_loss, test_loss

model_file_path              = 'situation_analysis/best_models/model_window60_horizon10.pth'
device                       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, val_loss, test_loss   = load_model(model_file_path, device)
print(f"Current val loss: {val_loss}")
pipeline                     = ContinualLearningPipeline(model, val_loss, test_loss)


# Example new data
data = pd.read_csv("data/JSE_clean_truncated.csv")
print(data.shape) # 3146 daily closing prices for 30 stocks
#data = data.head(1000) # minumum data points required 500
pipeline.simulate_noise = True  # Enable noise simulation
pipeline.continual_learning_step(data)
