import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime, timedelta

DB_CONFIG = {
    "drivername": "postgresql",
    "username": "postgres",
    "password": "darpen",  
    "host": "localhost",
    "port": 5433,  
    "database": "Electricity_UsageMonitoring",
}

engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

query = """
SELECT timestamp_start AS prediction_timestamp, 
       total_load_mwh AS actual_usage, 
       temperature_c, 
       humidity_percent, 
       wind_speed_mps
FROM public.electricty_weather;
"""
df = pd.read_sql_query(query, engine)

if df.empty:
    print("❌ No data fetched from the database! Please check your table or query.")
    exit()

df['id'] = range(1, len(df) + 1)  
print("✅ Data fetched and IDs assigned successfully!")

pd.set_option('future.no_silent_downcasting', True)
df.fillna(0, inplace=True)

df['rolling_mean'] = df['actual_usage'].rolling(window=5, min_periods=1).mean()
df['rolling_std'] = df['actual_usage'].rolling(window=5, min_periods=1).std()
df['lag_1'] = df['actual_usage'].shift(1)
df['lag_2'] = df['actual_usage'].shift(2)

df['temp_humidity'] = pd.to_numeric(df['temperature_c'] * df['humidity_percent'], errors='coerce')
df['wind_temp'] = pd.to_numeric(df['wind_speed_mps'] * df['temperature_c'], errors='coerce')

df.fillna(0, inplace=True)

X = df[['rolling_mean', 'rolling_std', 'lag_1', 'lag_2', 'temp_humidity', 'wind_temp']]
y = df['actual_usage']

print("Data types of feature set:")
print(X.dtypes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)

predictions = xgb_model.predict(X_test)

predicted_timestamps = pd.date_range(
    start=datetime.now(),  
    periods=len(y_test),
    freq="T"  
).tolist()

predictions_df = pd.DataFrame({
    "id": range(1, len(predicted_timestamps) + 1),  
    "prediction_timestamp": predicted_timestamps,
    "predicted_usage": predictions,
    "model_name": "XGBoost",
    "actual_value": y_test.values
})

try:
    with engine.begin() as connection:
        predictions_df.to_sql("backup_predicted_values", connection, if_exists="append", index=False)
    print("✅ Predictions stored successfully in the backup table `backup_predicted_values`!")
except Exception as e:
    print(f"❌ Error storing predictions: {e}")

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"✅ Mean Absolute Error (MAE): {mae}")
print(f"✅ Mean Squared Error (MSE): {mse}")
print(f"✅ R² Score: {r2}")

print(f"Sample of Predictions Stored in Backup Table:\n{predictions_df.head()}")