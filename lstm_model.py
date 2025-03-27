import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
 
# 📌 Database Configuration
DB_CONFIG = {
    "drivername": "postgresql",
    "username": "postgres",
    "password": "darpen",  
    "host": "localhost",
    "port": 5433,
    "database": "Electricity_UsageMonitoring",
}
 
# ✅ Connect to the database
engine = create_engine(
    f"postgresql+psycopg2://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)
 
# 📌 Fetch data
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
 
print("✅ Data fetched successfully!")
 
pd.set_option('future.no_silent_downcasting', True)
df.fillna(0, inplace=True)
 
# 📊 Feature Engineering
df['rolling_mean'] = df['actual_usage'].rolling(window=5, min_periods=1).mean()
df['rolling_std'] = df['actual_usage'].rolling(window=5, min_periods=1).std()
df['lag_1'] = df['actual_usage'].shift(1)
df['lag_2'] = df['actual_usage'].shift(2)
df['temp_humidity'] = pd.to_numeric(df['temperature_c'] * df['humidity_percent'], errors='coerce')
df['wind_temp'] = pd.to_numeric(df['wind_speed_mps'] * df['temperature_c'], errors='coerce')
df.fillna(0, inplace=True)
 
# 🔄 Scaling
feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(df[['rolling_mean', 'rolling_std', 'lag_1', 'lag_2', 'temp_humidity', 'wind_temp']])
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(df[['actual_usage']])
X = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
y = y_scaled.flatten()
 
# 🔀 Train-Test Split
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
 
# 🧠 LSTM Model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=25, batch_size=64, verbose=1)
 
# 🔮 Prediction
predictions = model.predict(X_test)
predictions = target_scaler.inverse_transform(predictions)
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))
 
# 📅 Generate Timestamps
predicted_timestamps = pd.date_range(start=datetime.now(), periods=len(y_test_actual), freq="T")
 
# 🧾 Build DataFrame
predictions_df = pd.DataFrame({
    "prediction_timestamp": predicted_timestamps,
    "predicted_usage": predictions.flatten(),
    "model_name": "LSTM",
    "actual_value": y_test_actual.flatten()
})
 
# 🛢️ Store in DB (dropping 'id' to match table schema)
try:
    with engine.begin() as connection:
        predictions_df.to_sql("backup_predicted_values", connection, if_exists="append", index=False)
    print("✅ Predictions successfully stored in `backup_predicted_values`!")
except Exception as e:
    print(f"❌ Error during database insertion: {e}")
 
# 📈 Evaluation Metrics
mae = mean_absolute_error(y_test_actual, predictions)
mse = mean_squared_error(y_test_actual, predictions)
r2 = r2_score(y_test_actual, predictions)
 
print(f"✅ MAE: {mae:.2f}")
print(f"✅ MSE: {mse:.2f}")
print(f"✅ R²: {r2:.2f}")
 
# 🧪 Show Sample
print(f"\nSample Predictions:\n{predictions_df.head()}")