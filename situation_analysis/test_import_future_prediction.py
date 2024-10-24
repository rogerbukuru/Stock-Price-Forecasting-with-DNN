
from stock_predictor import get_stock_predictions
import pandas as pd

# Get predictions
predictions_df, detailed_results, predictions_path = get_stock_predictions('../data/INVEST_GNN_clean.csv')

if predictions_df is not None:
    # Use the DataFrame directly
    print("Predictions DataFrame:")
    print(predictions_df)
    

else:
    print("Failed to get predictions")