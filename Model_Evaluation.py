import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

DB_CONFIG = {
    "drivername": "postgresql",
    "username": "postgres",
    "password": "darpen", 
    "host": "localhost",
    "port": 5433,
    "database": "Electricity_UsageMonitoring",
}

DATABASE_URL = URL.create(**DB_CONFIG)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

query = "SELECT * FROM backup_predicted_values;"  
df = pd.read_sql(query, engine)

if df.empty:
    print("❌ No data found in backup_predicted_values table!")
    exit()

print("✅ Data fetched successfully!")
print(df.head())

xgb_predictions = df[df["model_name"] == "XGBoost"]
lstm_predictions = df[df["model_name"] == "LSTM"]

def validate_predictions(predictions, model_name):
    if predictions.empty:
        print(f"❌ No predictions found for model: {model_name}.")
        return None
    
    if predictions.isnull().any().any():
        print(f"⚠️ Predictions for {model_name} contain NaN values. Filling NaNs with 0.")
        predictions = predictions.fillna(0)

    return predictions

xgb_predictions = validate_predictions(xgb_predictions, "XGBoost")
lstm_predictions = validate_predictions(lstm_predictions, "LSTM")

if xgb_predictions is None or lstm_predictions is None:
    print("❌ Unable to proceed due to missing or invalid predictions!")
    exit()

y_xgb_pred = xgb_predictions["predicted_usage"]
y_lstm_pred = lstm_predictions["predicted_usage"]

if y_xgb_pred.empty or y_lstm_pred.empty:
    print("❌ Predictions for one or both models are empty after validation!")
    exit()

def summarize_predictions(y_pred, model_name):
    if y_pred.empty:
        print(f"❌ Model {model_name} has no valid predictions.")
        return None

    mean_pred = np.mean(y_pred)
    std_pred = np.std(y_pred)
    min_pred = np.min(y_pred)
    max_pred = np.max(y_pred)
    median_pred = np.median(y_pred)

    print(f"\n📊 Model: {model_name}")
    print(f"➡️ Mean Predicted Usage: {mean_pred:.4f}")
    print(f"➡️ Standard Deviation: {std_pred:.4f}")
    print(f"➡️ Min Predicted Value: {min_pred:.4f}")
    print(f"➡️ Max Predicted Value: {max_pred:.4f}")
    print(f"➡️ Median Predicted Value: {median_pred:.4f}")
    print("-" * 50)

    return {
        "mean": mean_pred,
        "std": std_pred,
        "min": min_pred,
        "max": max_pred,
        "median": median_pred,
    }

xgb_summary = summarize_predictions(y_xgb_pred, "XGBoost")
lstm_summary = summarize_predictions(y_lstm_pred, "LSTM")

if not xgb_summary or not lstm_summary:
    print("❌ Unable to complete model evaluation due to invalid data!")
    exit()

print("\n✅ Model Evaluation & Comparison Completed Successfully!")