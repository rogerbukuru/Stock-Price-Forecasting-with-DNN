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

class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        Forward pass with correct dimension handling
        Args:
            x: Input tensor [batch_size, seq_length, features]
            adj: Adjacency matrix [num_nodes, num_nodes]
        """
        batch_size, seq_length, _ = x.size()
        x_reshaped = x.reshape(batch_size * seq_length, -1)
        out = self.fc(x_reshaped)
        out = out.view(batch_size, seq_length, -1)
        return out

class GraphWaveNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_nodes, dropout_rate):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Initialize adjacency matrix
        self.adapt_adj = torch.nn.Parameter(torch.eye(num_nodes))
        
        # Graph convolution layers
        self.gc1 = GraphConvLayer(input_size, hidden_size)
        self.gc2 = GraphConvLayer(hidden_size, hidden_size)
        
        # Temporal convolution
        self.temporal_conv = torch.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        
        # Output layer
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.gc1(x, self.adapt_adj)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.gc2(x, self.adapt_adj)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = torch.relu(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        
        x = x[:, -1, :]
        out = self.fc(x)
        
        return out

class DataValidator:
    def __init__(self, expected_columns, window_size):
        self.expected_columns = expected_columns
        self.window_size = window_size
    
    def validate_data_format(self, data):
        try:
            if data.empty:
                return False, "Input data is empty"
            
            missing_cols = set(self.expected_columns) - set(data.columns)
            if missing_cols:
                return False, f"Missing columns: {missing_cols}"
            
            if len(data) < self.window_size:
                return False, f"Need at least {self.window_size} rows, got {len(data)}"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def preprocess_data(self, data):
        try:
            is_valid, message = self.validate_data_format(data)
            if not is_valid:
                return False, message
            
            processed_data = data.tail(self.window_size).copy()
            processed_data = processed_data[self.expected_columns]
            
            return True, processed_data
            
        except Exception as e:
            return False, f"Preprocessing error: {str(e)}"

class StockPredictor:
    def __init__(self, model_path='best_models/model_window30_horizon1.pth', 
                 train_data_path='../data/INVEST_GNN_clean.csv'):
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Extract model configuration first
            logger.info("Extracting model configuration...")
            self.model_config = self.extract_model_config(model_path)
            logger.info("Model configuration extracted:")
            for key, value in self.model_config.items():
                logger.info(f"{key}: {value}")
            
            # Load training data
            logger.info("Loading training data...")
            self.load_training_data(train_data_path)
            
            # Initialize model with extracted configuration
            logger.info("Initializing model...")
            self.initialize_model()
            
            # Initialize validator with extracted window size
            self.validator = DataValidator(self.feature_names, self.model_config['window_size'])
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def extract_model_config(self, model_path):
        """Extract model configuration from saved checkpoint"""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize a temporary small model to get the state dict structure
        temp_model = GraphWaveNet(
            input_size=1,  # Temporary size
            hidden_size=1,  # Temporary size
            output_size=1,  # Temporary size
            num_nodes=1,    # Temporary size
            dropout_rate=0.1
        )
        
        # Get the state dict structure
        state_dict = checkpoint['model_state_dict']
        
        # Extract parameters from state dict
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
        
        return config

    def load_training_data(self, train_data_path):
        """Load and prepare training data with robust data cleaning"""
        try:
            # Load raw data
            self.train_data = pd.read_csv(train_data_path)
            logger.info(f"Raw data shape: {self.train_data.shape}")
            
            # Store original column names
            self.feature_names = list(self.train_data.columns)
            
            # Data cleaning steps
            for column in self.feature_names:
                # Convert to string first to handle any potential mixed types
                self.train_data[column] = pd.to_numeric(
                    self.train_data[column].astype(str).str.replace(',', ''),  # Remove commas
                    errors='coerce'  # Convert errors to NaN
                )
            
            # Check for any remaining non-numeric values
            non_numeric_cols = self.train_data.select_dtypes(exclude=['float64', 'int64']).columns
            if len(non_numeric_cols) > 0:
                raise ValueError(f"Non-numeric values found in columns: {list(non_numeric_cols)}")
            
            # Check for NaN values
            nan_cols = self.train_data.columns[self.train_data.isna().any()].tolist()
            if nan_cols:
                logger.warning(f"NaN values found in columns: {nan_cols}")
                # Forward fill followed by backward fill
                self.train_data = self.train_data.fillna(method='ffill').fillna(method='bfill')
            
            # Check for infinite values
            inf_cols = self.train_data.columns[np.isinf(self.train_data).any()].tolist()
            if inf_cols:
                logger.warning(f"Infinite values found in columns: {inf_cols}")
                # Replace infinite values with NaN and then forward/backward fill
                self.train_data = self.train_data.replace([np.inf, -np.inf], np.nan)
                self.train_data = self.train_data.fillna(method='ffill').fillna(method='bfill')
            
            # Verify all data is numeric and finite
            if not np.isfinite(self.train_data.values).all():
                raise ValueError("Data contains non-finite values after cleaning")
            
            logger.info("Data cleaning completed successfully")
            logger.info(f"Cleaned data shape: {self.train_data.shape}")
            
            # Initialize and fit the scaler
            self.scaler = StandardScaler()
            self.scaler.fit(self.train_data)
            
        except Exception as e:
            logger.error(f"Error in data loading and cleaning: {str(e)}")
            raise

    def initialize_model(self):
        """Initialize model with extracted configuration"""
        num_features = len(self.feature_names)
        self.model = GraphWaveNet(
            input_size=num_features,
            hidden_size=self.model_config['hidden_size'],
            output_size=num_features,
            num_nodes=self.model_config['num_nodes'],
            dropout_rate=self.model_config['dropout_rate']
        ).to(self.device)
        
        # Load the state dict
        checkpoint = torch.load(self.model_config['model_path'], map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def validate_input_data(self, input_data):
        """Validate and clean input data using the same process as training data"""
        try:
            cleaned_data = input_data.copy()
            
            # Apply same cleaning steps as training data
            for column in self.feature_names:
                cleaned_data[column] = pd.to_numeric(
                    cleaned_data[column].astype(str).str.replace(',', ''),
                    errors='coerce'
                )
            
            # Handle NaN values
            if cleaned_data.isna().any().any():
                cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
            
            # Handle infinite values
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
            cleaned_data = cleaned_data.fillna(method='ffill').fillna(method='bfill')
            
            # Verify all data is numeric and finite
            if not np.isfinite(cleaned_data.values).all():
                raise ValueError("Input data contains non-finite values after cleaning")
            
            return True, cleaned_data
            
        except Exception as e:
            return False, f"Data validation failed: {str(e)}"

    def predict(self, input_data):
        try:
            # Validate and clean input data first
            is_valid, cleaned_data = self.validate_input_data(input_data)
            if not is_valid:
                logger.error(cleaned_data)  # cleaned_data contains error message in this case
                return None, None
            
            horizon = self.model_config['prediction_horizon']
            
            # Validate and preprocess input data
            is_valid, result = self.validator.preprocess_data(cleaned_data)
            if not is_valid:
                logger.error(f"Data validation failed: {result}")
                return None, None
            
            processed_data = result
            
            # Scale data
            scaled_data = self.scaler.transform(processed_data)
            x = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
            
            logger.info(f"Input shape: {x.shape}")
            
            # Make prediction
            with torch.no_grad():
                predicted_scaled = self.model(x)
                logger.info(f"Output shape: {predicted_scaled.shape}")
                predicted = self.scaler.inverse_transform(
                    predicted_scaled.cpu().numpy().reshape(1, -1)
                )
            
            # Create results DataFrame
            predicted_prices = pd.DataFrame([predicted.squeeze()], columns=self.feature_names)
            
            # Create detailed results
            last_known_prices = processed_data.iloc[-1]
            percent_changes = ((predicted.squeeze() - last_known_prices) / last_known_prices * 100).round(2)
            
            detailed_results = pd.DataFrame({
                'Stock': self.feature_names,
                'Last_Price': last_known_prices,
                f'Predicted_Price_{horizon}_day_ahead': predicted.squeeze(),
                'Percent_Change': percent_changes
            })
            
            detailed_results = detailed_results.sort_values('Percent_Change', ascending=False)
            
            # Create visualization
            self.visualize_predictions(processed_data, detailed_results, horizon)
            
            # Save results
            output_path = 'predictions'
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            detailed_results.to_csv(os.path.join(output_path, f'detailed_predictions_{timestamp}.csv'))
            
            logger.info("Prediction successful")
            return predicted_prices, detailed_results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None, None

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
                    f'Predicted_Price_{horizon}_day_ahead'
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
            
            # Remove empty subplots if any
            for idx in range(n_stocks, len(axes)):
                fig.delaxes(axes[idx])
                
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(plot_dir, f'predictions_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Prediction plot saved to {plot_path}")
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")

def get_stock_predictions(data_path):
    """
    Get stock predictions using the extracted model configuration.
    
    Args:
        data_path (str): Path to input data CSV file
        
    Returns:
        tuple: (predictions_df, detailed_results, csv_path)
            - predictions_df: DataFrame containing predictions
            - detailed_results: DataFrame with detailed analysis
            - csv_path: Path to saved predictions CSV file
    """
    try:
        # Make predictions
        predictor = StockPredictor()
        data = pd.read_csv(data_path)
        predictions, detailed_results = predictor.predict(data)
        
        if predictions is not None:
            # Create output directory
            output_path = 'predictions'
            os.makedirs(output_path, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            predictions_filename = f'predictions_{timestamp}.csv'
            csv_path = os.path.join(output_path, predictions_filename)
            
            # Save predictions
            predictions.to_csv(csv_path, index=False)
            
            logger.info(f"Predictions saved to {csv_path}")
            return predictions, detailed_results, csv_path
            
        return None, None, None
        
    except Exception as e:
        logger.error(f"Error getting predictions: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    try:
        data_path = '../data/INVEST_GNN_clean.csv'
        predictions, detailed_results, predictions_path = get_stock_predictions(data_path)
        
        if predictions is not None:
            print("\nPredicted Prices:")
            print(predictions)
            
            print("\nDetailed Results:")
            print(detailed_results)
            
            print(f"\nPredictions saved to: {predictions_path}")
        else:
            print("Failed to get predictions")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")