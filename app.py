import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import plot_importance
 
# 🎨 Streamlit Page Config
st.set_page_config(page_title="Electricity Usage Dashboard", layout="wide")
 
# 📌 Database Configuration
DB_CONFIG = {
    "drivername": "postgresql",
    "username": "postgres",
    "password": "darpen",  # Replace securely in production
    "host": "localhost",
    "port": 5433,
    "database": "Electricity_UsageMonitoring",
}
 
# 📌 Function to Fetch Data
@st.cache_data
def get_data_from_database(query):
    engine = create_engine(
        f"{DB_CONFIG['drivername']}://{DB_CONFIG['username']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return pd.read_sql(query, con=engine)
 
# 📌 Load Models
@st.cache_resource
def load_models():
    xgb_model = pickle.load(open("Models/xgboost_model.pkl", "rb"))
    lstm_model = load_model("Models/lstm_model.h5")
    return xgb_model, lstm_model
 
xgb_model, lstm_model = load_models()
 
# 🌐 Sidebar: Model Selection
st.sidebar.title("🔹 Filters")
model_selected = st.sidebar.selectbox("📊 Select Model", ["XGBoost", "LSTM"])
 
# 📌 Fetch All Data
st.write("### Loading Data...")
 
renewable_query = """
SELECT utc_timestamp, de_load_actual_entsoe_transparency, de_solar_capacity
FROM renewable_data;
"""
renewable_data = get_data_from_database(renewable_query)
 
electricity_query = """
SELECT timestamp_start, total_load_mwh, temperature_c, humidity_percent
FROM electricty_weather;
"""
electricity_data = get_data_from_database(electricity_query)
 
# Merge and prepare data
merged_data = electricity_data.merge(
    renewable_data,
    left_on='timestamp_start',
    right_on='utc_timestamp',
    how='left'
)
 
merged_data['year'] = pd.to_datetime(merged_data['timestamp_start']).dt.year
 
# If solar_capacity is missing, fill with simulated values
if merged_data['de_solar_capacity'].isnull().all():
    st.warning("⚠️ Renewable data not found. Showing simulated values.")
    merged_data['de_solar_capacity'] = np.random.randint(1000, 10000, size=len(merged_data))
 
# Renewable contribution
merged_data['renewable_contribution (%)'] = (merged_data['de_solar_capacity'] / merged_data['total_load_mwh']) * 100
merged_data.interpolate(inplace=True)
 
# Prepare X and y
X_test = merged_data[['temperature_c', 'humidity_percent', 'de_solar_capacity']].apply(pd.to_numeric, errors='coerce').fillna(0)
y_test = merged_data['total_load_mwh']
 
# 📌 Prediction
if model_selected == "XGBoost":
    y_pred = xgb_model.predict(X_test)
elif model_selected == "LSTM":
    try:
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred = lstm_model.predict(X_test_lstm).flatten()
    except ValueError as e:
        st.error(f"LSTM model failed due to invalid input shape or dtype: {e}")
        y_pred = np.zeros(X_test.shape[0])
 
# 📌 Evaluation
try:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
 
    # ✅ Correct percentage based on mean
    y_mean = y_test.mean()
    rmse_percent = (rmse / y_mean) * 100
    mae_percent = (mae / y_mean) * 100
except ValueError as e:
    st.error(f"Error in evaluation metrics calculation: {e}")
    mse = rmse = r2 = mae = rmse_percent = mae_percent = np.nan
 
# 🎯 Dashboard Title
st.markdown("<h1 style='text-align: center; color: cyan;'>⚡ Electricity & Renewable Usage Dashboard</h1>", unsafe_allow_html=True)
 
# 🔹 Key Metrics
col1, col2, col3, col4 = st.columns(4)
avg_renew = merged_data['renewable_contribution (%)'].mean()
col1.metric("🔋 Avg. Renewable Contribution (%)", f"{avg_renew:.2f}%" if not np.isnan(avg_renew) else "Simulated")
col2.metric("📌 Selected Model", model_selected)
col3.metric(f"📊 {model_selected} RMSE (%)", f"{rmse_percent:.2f}%" if not np.isnan(rmse_percent) else "N/A")
col4.metric(f"📉 {model_selected} MAE (%)", f"{mae_percent:.2f}%" if not np.isnan(mae_percent) else "N/A")
 
# 🔹 Bar Charts
col1, col2 = st.columns(2)
fig1 = px.bar(merged_data[:20], x="timestamp_start", y="renewable_contribution (%)", title="Renewable Contribution (%) Over Time", color_discrete_sequence=['green'])
fig1.update_layout(xaxis_tickangle=-45)
col1.plotly_chart(fig1, use_container_width=True)
 
fig2 = px.bar(
    pd.DataFrame({
        "Index": range(len(y_test[:20])),
        "Actual Usage": y_test[:20].values,
        "Predicted Usage": y_pred[:20]
    }),
    x="Index",
    y=["Actual Usage", "Predicted Usage"],
    barmode="group",
    title="Actual vs Predicted Electricity Usage"
)
col2.plotly_chart(fig2, use_container_width=True)
 
# 🔹 Line Graphs
col3, col4 = st.columns(2)
fig3 = px.line(merged_data, x="year", y="total_load_mwh", title="Yearly Electricity Usage Trend", markers=True)
col3.plotly_chart(fig3, use_container_width=True)
 
fig4 = px.line(
    x=range(len(y_test[:50])),
    y=[y_test[:50], y_pred[:50]],
    labels={"x": "Index", "value": "Usage (MWh)"},
    title="Actual vs Predicted Usage Over Time"
)
fig4.update_layout(legend=dict(title="Legend", itemsizing='constant'))
col4.plotly_chart(fig4, use_container_width=True)
 
# 🔹 Insights
col5, col6 = st.columns(2)
fig5 = px.pie(
    names=["Renewable", "Non-Renewable"],
    values=[avg_renew, 100 - avg_renew] if not np.isnan(avg_renew) else [50, 50],
    title="Renewable Contribution (%)",
    color_discrete_sequence=["green", "gray"]
)
col5.plotly_chart(fig5, use_container_width=True)
 
fig6, ax6 = plt.subplots(figsize=(5, 4))
sns.heatmap(merged_data[['total_load_mwh', 'temperature_c', 'humidity_percent', 'de_solar_capacity']].corr(), annot=True, cmap="coolwarm", ax=ax6, fmt=".2f")
ax6.set_title("Correlation Heatmap")
col6.pyplot(fig6)
 
# 🔍 Feature Importance (XGBoost only)
if model_selected == "XGBoost":
    st.write("### 🔍 Feature Importance (XGBoost)")
    fig_imp, ax_imp = plt.subplots()
    plot_importance(xgb_model, ax=ax_imp)
    st.pyplot(fig_imp)
 
# 📌 Model Summary
st.write("## 🚀 Model Performance Comparison")
st.write(
    f"**{model_selected}**: RMSE = {rmse:,.2f} ({rmse_percent:.2f}%), "
    f"MAE = {mae:,.2f} ({mae_percent:.2f}%), "
    f"R² = {r2:.2f}" if not np.isnan(rmse) else f"**{model_selected}**: Metrics unavailable"
)
 
# Footer
st.markdown("<h5 style='text-align: center;'>Developed by Darpen Bhandari</h5>", unsafe_allow_html=True)