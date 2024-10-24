import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Best model parameters from training
MODEL_PARAMS = {
    'input_window': 60,
    'prediction_horizon': 10,
    'hidden_size': 128,
    'learning_rate': 0.01,
    'dropout_rate': 0.2,
    'num_nodes': 60
}

class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = self.fc(out)
        return out

class GraphWaveNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_nodes, dropout_rate=0.2):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.adapt_adj = torch.nn.Parameter(torch.eye(num_nodes))
        self.gc1 = GraphConvLayer(input_size, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        self.temporal_conv = torch.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

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

class DataValidator:
    def __init__(self, expected_columns):
        self.expected_columns = expected_columns
        self.input_window = MODEL_PARAMS['input_window']
    
    def validate_data_format(self, data):
        try:
            if data.empty:
                return False, "Input data is empty"
            
            missing_cols = set(self.expected_columns) - set(data.columns)
            if missing_cols:
                return False, f"Missing columns: {missing_cols}"
            
            non_numeric = data.select_dtypes(exclude=['float64', 'int64']).columns
            if not non_numeric.empty:
                return False, f"Non-numeric columns found: {list(non_numeric)}"
            
            if data.isnull().any().any():
                return False, "Data contains missing values"
            
            if len(data) < self.input_window:
                return False, f"Need at least {self.input_window} rows, got {len(data)}"
            
            if (data < 0).any().any():
                return False, "Negative values found in data"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def preprocess_data(self, data):
        try:
            is_valid, message = self.validate_data_format(data)
            if not is_valid:
                return False, message
            
            processed_data = data.tail(self.input_window).copy()
            processed_data = processed_data[self.expected_columns]
            
            return True, processed_data
            
        except Exception as e:
            return False, f"Preprocessing error: {str(e)}"

class StockPredictor:
    def __init__(self, model_path='best_models/model_window60_horizon10.pth', 
                 train_data_path='data/JSE_clean_truncated.csv'):
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load training data
            logger.info("Loading training data...")
            self.load_training_data(train_data_path)
            
            # Initialize model
            logger.info("Initializing model...")
            self.initialize_model()
            
            # Load model weights
            logger.info("Loading model weights...")
            self.load_model_weights(model_path)
            
            # Initialize validator
            self.validator = DataValidator(self.feature_names)
            
            logger.info("Initialization complete!")
            logger.info(f"Using model parameters:")
            for key, value in MODEL_PARAMS.items():
                logger.info(f"{key}: {value}")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def load_training_data(self, train_data_path):
        self.train_data = pd.read_csv(train_data_path)
        self.feature_names = list(self.train_data.columns)
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_data)

    def initialize_model(self):
        self.model = GraphWaveNet(
            input_size=len(self.feature_names),
            hidden_size=MODEL_PARAMS['hidden_size'],
            output_size=len(self.feature_names),
            num_nodes=MODEL_PARAMS['num_nodes'],
            dropout_rate=MODEL_PARAMS['dropout_rate']
        ).to(self.device)

    def load_model_weights(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, input_data, horizon=None):
        try:
            if horizon is None:
                horizon = MODEL_PARAMS['prediction_horizon']
            
            # Validate and preprocess input data
            is_valid, result = self.validator.preprocess_data(input_data)
            if not is_valid:
                logger.error(f"Data validation failed: {result}")
                return None
            
            processed_data = result
            
            # Scale data
            scaled_data = self.scaler.transform(processed_data)
            x = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                predicted_scaled = self.model(x)
                predicted = self.scaler.inverse_transform(
                    predicted_scaled.cpu().numpy().reshape(1, -1)
                )
            
            # Create results DataFrame in original format
            predicted_prices = pd.DataFrame([predicted.squeeze()], columns=self.feature_names)
            
            # Create detailed results for visualization and analysis
            last_known_prices = processed_data.iloc[-1]
            percent_changes = ((predicted.squeeze() - last_known_prices) / last_known_prices * 100).round(2)
            
            detailed_results = pd.DataFrame({
                'Stock': self.feature_names,
                'Last_Price': last_known_prices,
                f'Predicted_Price_{horizon}_steps_ahead': predicted.squeeze(),
                'Percent_Change': percent_changes
            })
            
            # Sort detailed results by percent change
            detailed_results = detailed_results.sort_values('Percent_Change', ascending=False)
            
            # Create visualization
            self.visualize_predictions(processed_data, detailed_results, horizon)
            
            # Save detailed results
            output_path = 'predictions'
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            detailed_results.to_csv(os.path.join(output_path, f'detailed_predictions_{timestamp}.csv'))
            
            logger.info("Prediction successful")
            return predicted_prices
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None

    def visualize_predictions(self, historical_data, detailed_results, horizon):
        try:
            plot_dir = 'prediction_plots'
            os.makedirs(plot_dir, exist_ok=True)
            
            plt.style.use('seaborn')
            n_stocks = len(self.feature_names)
            n_cols = 3
            n_rows = (n_stocks + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            axes = axes.flatten()
            
            for idx, stock in enumerate(self.feature_names):
                ax = axes[idx]
                
                # Plot historical data
                historical = historical_data[stock].values
                ax.plot(range(-len(historical), 0), historical, 
                       label='Historical', color='blue', marker='o')
                
                # Plot prediction
                predicted = detailed_results.loc[
                    detailed_results['Stock'] == stock,
                    f'Predicted_Price_{horizon}_steps_ahead'
                ].values[0]
                ax.scatter(horizon, predicted, color='red', s=100, 
                         label='Prediction', marker='*')
                
                # Add percentage change annotation
                pct_change = detailed_results.loc[
                    detailed_results['Stock'] == stock,
                    'Percent_Change'
                ].values[0]
                color = 'green' if pct_change >= 0 else 'red'
                ax.annotate(f'{pct_change:.1f}%', 
                          xy=(horizon, predicted),
                          xytext=(5, 5), textcoords='offset points',
                          color=color, fontweight='bold')
                
                # Add current price to title
                current_price = detailed_results.loc[
                    detailed_results['Stock'] == stock,
                    'Last_Price'
                ].values[0]
                ax.set_title(f"{stock}\nCurrent: {current_price:.2f}")
                ax.grid(True)
                ax.legend()
                
                # Set x-axis labels
                ax.set_xticks(list(range(-len(historical), horizon+1, 5)))
                
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(plot_dir, f'predictions_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Prediction plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")

def predict_stock_prices(data_path, horizon=None):
    """Convenience function to load data and make predictions."""
    try:
        predictor = StockPredictor()
        data = pd.read_csv(data_path)
        predictions = predictor.predict(data, horizon=horizon)
        return predictions
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        # Using best model's prediction horizon by default
        predictions = predict_stock_prices('data/JSE_clean_truncated.csv')
        
        if predictions is not None:
            print("\nPredicted Prices:")
            print(predictions)
            
            # Save predictions in original format
            output_path = 'predictions'
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            
            # Save predictions
            filename = f'predictions_{timestamp}.csv'
            predictions.to_csv(os.path.join(output_path, filename), index=False)
            logger.info(f"Predictions saved to {os.path.join(output_path, filename)}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")