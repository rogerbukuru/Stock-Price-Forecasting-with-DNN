import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import json

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Loading and Preprocessing
def load_and_preprocess_data(jse_file, invest_file, sector_files):
    # Load JSE price data
    jse_data = pd.read_csv(jse_file)
    
    # Load INVEST fundamental data
    invest_data = pd.read_csv(invest_file, parse_dates=['Date'])
    invest_data.set_index('Date', inplace=True)

    # Load sector-specific stocks
    sectors = {}
    for sector, file in sector_files.items():
        with open(file, 'r') as f:
            sectors[sector] = json.load(f)

    # Prepare JSE data
    scaler = StandardScaler()
    scaled_jse_data = pd.DataFrame(scaler.fit_transform(jse_data), columns=jse_data.columns)

    return jse_data, scaled_jse_data, invest_data, scaler, sectors

# Create sequences for time series forecasting
def create_sequences(data, seq_length, horizon):
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon + 1):
        x = data[i:i + seq_length]
        y = data[i + seq_length + horizon - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

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

# Continuous Learning Component
class ContinuousLearningComponent:
    def __init__(self, input_size, hidden_size, output_size, num_nodes, learning_rate=0.001, dropout_rate=0.2):
        self.model = GraphWaveNet(input_size, hidden_size, output_size, num_nodes, dropout_rate).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    def train(self, train_data, val_data, epochs):
        self.model.train()
        train_losses, val_losses = [], []
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.optimizer.zero_grad()
            outputs = self.model(train_data)
            loss_fn = nn.MSELoss()
            loss = loss_fn(outputs, train_data[:, -1, :])  # Compare with the last time step
            loss.backward()
            self.optimizer.step()
            
            val_loss = self.validate(val_data)
            self.scheduler.step(val_loss)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        
        return train_losses, val_losses

    def validate(self, val_data):
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(val_data)
            val_loss_fn = nn.MSELoss()
            val_loss = val_loss_fn(val_outputs, val_data[:, -1, :])  # Compare with the last time step
        return val_loss.item()

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

# Decision-Making Component
class DecisionMakingComponent:
    def __init__(self):
        self.bn = gum.BayesNet('Investment Decision')
        self.setup_network()

    def setup_network(self):
        # Add nodes (modify as per INVEST system structure)
        self.bn.add(gum.LabelizedVariable('price_trend', 'Price Trend', ['decrease', 'stable', 'increase']))
        self.bn.add(gum.LabelizedVariable('pe_ratio', 'PE Ratio', ['low', 'medium', 'high']))
        self.bn.add(gum.LabelizedVariable('roae', 'ROAE', ['low', 'medium', 'high']))
        self.bn.add(gum.LabelizedVariable('debt_equity', 'Debt/Equity', ['low', 'medium', 'high']))
        self.bn.add(gum.LabelizedVariable('decision', 'Investment Decision', ['sell', 'hold', 'buy']))

        # Add arcs (modify as per INVEST system structure)
        self.bn.addArc('price_trend', 'decision')
        self.bn.addArc('pe_ratio', 'decision')
        self.bn.addArc('roae', 'decision')
        self.bn.addArc('debt_equity', 'decision')

    def discretize_data(self, data):
        discretized = data.copy()
        discretized['price_trend'] = pd.qcut(data['Price'].pct_change(), q=3, labels=['decrease', 'stable', 'increase'])
        discretized['pe_ratio'] = pd.qcut(data['PE'], q=3, labels=['low', 'medium', 'high'])
        discretized['roae'] = pd.qcut(data['ROAE'], q=3, labels=['low', 'medium', 'high'])
        discretized['debt_equity'] = pd.qcut(data['Debt/Equity'], q=3, labels=['low', 'medium', 'high'])
        return discretized

    def train(self, training_data, algorithm='EM'):
        discretized_data = self.discretize_data(training_data)
        learner = gum.BNLearner(self.bn, discretized_data)
        
        if algorithm == 'EM':
            learner.useEM()
        elif algorithm == 'GreedyHillClimbing':
            learner.useGreedyHillClimbing()
        elif algorithm == 'LocalSearchWithTabuList':
            learner.useLocalSearchWithTabuList()
        
        self.bn = learner.learnParameters(discretized_data)

    def make_decision(self, evidence):
        ie = gum.LazyPropagation(self.bn)
        ie.setEvidence(evidence)
        ie.makeInference()
        decision_probs = ie.posterior('decision')
        return max(decision_probs.todict().items(), key=lambda x: x[1])[0]

    def walk_forward_validation(self, data, window_size, step_size, algorithms):
        results = {alg: [] for alg in algorithms}
        
        for i in range(0, len(data) - window_size, step_size):
            train_data = data.iloc[i:i+window_size]
            test_data = data.iloc[i+window_size:i+window_size+step_size]
            
            for alg in algorithms:
                self.train(train_data, algorithm=alg)
                
                test_discretized = self.discretize_data(test_data)
                predictions = []
                actual = test_discretized['decision'].tolist()
                
                for _, row in test_discretized.iterrows():
                    evidence = {
                        'price_trend': row['price_trend'],
                        'pe_ratio': row['pe_ratio'],
                        'roae': row['roae'],
                        'debt_equity': row['debt_equity']
                    }
                    predictions.append(self.make_decision(evidence))
                
                accuracy = (np.array(predictions) == np.array(actual)).mean()
                
                results[alg].append({
                    'window_start': data.index[i],
                    'window_end': data.index[i+window_size],
                    'accuracy': accuracy
                })
        
        return results

    def compare_algorithms(self, data, window_size=60, step_size=30):
        algorithms = ['EM', 'GreedyHillClimbing', 'LocalSearchWithTabuList']
        results = self.walk_forward_validation(data, window_size, step_size, algorithms)
        
        for alg in algorithms:
            df = pd.DataFrame(results[alg])
            print(f"Results for {alg}:")
            print(df.mean())
            print("\n")
        
        # Visualize results
        plt.figure(figsize=(12, 6))
        for alg in algorithms:
            df = pd.DataFrame(results[alg])
            plt.plot(df['window_end'], df['accuracy'], label=alg)
        plt.title('Algorithm Accuracy over Time')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('algorithm_comparison.png')
        plt.close()

    def show_bn(self):
        gnb.showBN(self.bn)

# Hybrid AI System
class HybridAISystem:
    def __init__(self, jse_file, invest_file, sector_files, input_window, prediction_horizon, hidden_size):
        self.jse_data, self.scaled_jse_data, self.invest_data, self.scaler, self.sectors = load_and_preprocess_data(jse_file, invest_file, sector_files)
        self.input_window = input_window
        self.prediction_horizon = prediction_horizon
        
        num_features = self.jse_data.shape[1]  # Number of stocks
        
        self.continuous_learning = ContinuousLearningComponent(
            input_size=1,  # Each node has one feature (price)
            hidden_size=hidden_size,
            output_size=num_features,
            num_nodes=num_features
        )
        self.decision_making = {sector: DecisionMakingComponent() for sector in self.sectors}

    def prepare_data_for_training(self):
        X, y = create_sequences(self.scaled_jse_data.values, self.input_window, self.prediction_horizon)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return (torch.FloatTensor(X_train).to(device),
                torch.FloatTensor(y_train).to(device),
                torch.FloatTensor(X_val).to(device),
                torch.FloatTensor(y_val).to(device))

    def train(self, epochs=100):
        # Train the Continuous Learning Component
        X_train, y_train, X_val, y_val = self.prepare_data_for_training()
        train_losses, val_losses = self.continuous_learning.train(X_train, X_val, epochs)
        
        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('continuous_learning_loss.png')
        plt.close()
        
        # Train the Decision-Making Component for each sector
        for sector, stocks in self.sectors.items():
            sector_data = self.invest_data[self.invest_data['Name'].isin(stocks)]
            self.decision_making[sector].compare_algorithms(sector_data)

    def make_investment_decision(self, current_date, sector):
        # Get the latest JSE data
        current_jse_data = self.jse_data.iloc[-self.input_window:]
        price_sequence = torch.FloatTensor(self.scaler.transform(current_jse_data)).unsqueeze(0).to(device)

        # Get price predictions
        price_predictions = self.continuous_learning.predict(price_sequence)
        price_predictions = self.scaler.inverse_transform(price_predictions.cpu().numpy())[0]

        # Get the latest INVEST data for the sector
        current_invest_data = self.invest_data[self.invest_data.index <= current_date].iloc[-1]

        # Calculate price trend
        current_prices = current_jse_data.iloc[-1].values
        price_trend = np.mean((price_predictions - current_prices) / current_prices)

        # Prepare evidence for decision making
        evidence = {
            'price_trend': 'increase' if price_trend > 0.01 else 'decrease' if price_trend < -0.01 else 'stable',
            'pe_ratio': self.decision_making[sector].discretize_data(pd.DataFrame({'PE': [current_invest_data['PE']]}))['pe_ratio'].iloc[0],
            'roae': self.decision_making[sector].discretize_data(pd.DataFrame({'ROAE': [current_invest_data['ROAE']]}))['roae'].iloc[0],
            'debt_equity': self.decision_making[sector].discretize_data(pd.DataFrame({'Debt/Equity': [current_invest_data['Debt/Equity']]}))['debt_equity'].iloc[0]
        }

        # Make decision
        decision = self.decision_making[sector].make_decision(evidence)
        return decision, price_predictions

    def run_simulation(self, start_date, end_date):
        simulation_results = {}
        for sector, stocks in self.sectors.items():
            print(f"Running simulation for {sector}...")
            sector_data = self.invest_data[self.invest_data['Name'].isin(stocks)]
            simulation_data = sector_data.loc[start_date:end_date]
            decisions = []
            predicted_prices = []
            actual_prices = []

            for date in tqdm(simulation_data.index):
                decision, price_prediction = self.make_investment_decision(date, sector)
                decisions.append(decision)
                predicted_prices.append(price_prediction)
                actual_prices.append(self.jse_data.iloc[len(self.jse_data) - len(simulation_data) + len(decisions) - 1][stocks].values)

            simulation_results[sector] = pd.DataFrame({
                'Date': simulation_data.index,
                'Decision': decisions,
                'Predicted_Prices': predicted_prices,
                'Actual_Prices': actual_prices
            })
        
        return simulation_results

    def evaluate_performance(self, simulation_results):
        for sector, results in simulation_results.items():
            print(f"\nPerformance Evaluation for {sector}:")
            
            # Calculate returns based on decisions
            returns = []
            for i in range(len(results)):
                if i == 0:
                    returns.append(0)
                else:
                    prev_prices = results['Actual_Prices'].iloc[i-1]
                    current_prices = results['Actual_Prices'].iloc[i]
                    price_change = (current_prices - prev_prices) / prev_prices
                    if results['Decision'].iloc[i-1] == 'buy':
                        returns.append(np.mean(price_change))
                    elif results['Decision'].iloc[i-1] == 'sell':
                        returns.append(-np.mean(price_change))
                    else:
                        returns.append(0)

            results['Returns'] = returns
            cumulative_returns = (1 + pd.Series(returns)).cumprod() - 1

            # Calculate performance metrics
            total_return = cumulative_returns.iloc[-1]
            average_annual_return = (1 + total_return) ** (252 / len(results)) - 1  # Assuming 252 trading days per year
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
            max_drawdown = np.min(cumulative_returns - cumulative_returns.cummax())

            print(f"Cumulative Return (CR): {total_return:.2%}")
            print(f"Average Annual Return (AAR): {average_annual_return:.2%}")
            print(f"Total Return (TR): {total_return:.2%}")
            print(f"Sharpe Ratio (SR): {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")

            # Plot cumulative returns
            plt.figure(figsize=(12, 6))
            cumulative_returns.plot()
            plt.title(f'Cumulative Returns for {sector}')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.savefig(f'cumulative_returns_{sector}.png')
            plt.close()

            # Share-level analysis
            share_returns = results['Actual_Prices'].pct_change().mean()
            share_contributions = share_returns * results['Decision'].map({'buy': 1, 'sell': -1, 'hold': 0}).mean()
            
            plt.figure(figsize=(12, 6))
            share_contributions.plot(kind='bar')
            plt.title(f'Share Contributions to AAR for {sector}')
            plt.xlabel('Shares')
            plt.ylabel('Contribution to AAR')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'share_contributions_{sector}.png')
            plt.close()

            print("\nTop 5 Contributing Shares:")
            print(share_contributions.nlargest(5))
            print("\nBottom 5 Contributing Shares:")
            print(share_contributions.nsmallest(5))

    def analyze_cpt_learning(self):
        for sector in self.sectors:
            print(f"\nCPT Learning Analysis for {sector}:")
            sector_data = self.invest_data[self.invest_data['Name'].isin(self.sectors[sector])]
            self.decision_making[sector].compare_algorithms(sector_data)

# Main execution
if __name__ == "__main__":
    hybrid_system = HybridAISystem(
        jse_file='data/JSE_clean_truncated.csv',
        invest_file='data/INVEST_clean.csv',
        sector_files={'JCSEV': 'data/jcsev.json', 'JGIND': 'data/jgind.json'},
        input_window=60,
        prediction_horizon=5,
        hidden_size=64
    )

    print("Training the Hybrid AI System...")
    hybrid_system.train(epochs=100)

    print("Running simulation...")
    simulation_results = hybrid_system.run_simulation('2015-01-01', '2018-12-31')

    print("Evaluating performance...")
    hybrid_system.evaluate_performance(simulation_results)

    print("Analyzing CPT learning algorithms...")
    hybrid_system.analyze_cpt_learning()

    print("Simulation complete. Check the generated plots and results for detailed analysis.")

    # Save simulation results
    for sector, results in simulation_results.items():
        results.to_csv(f'simulation_results_{sector}.csv', index=False)

    # Display Bayesian Network structure for each sector
    for sector, decision_making in hybrid_system.decision_making.items():
        decision_making.show_bn()
        plt.savefig(f'bayesian_network_{sector}.png')
        plt.close()