⚡ Electricity & Renewable Energy Usage Insights Dashboard
This project presents an end-to-end data science solution to monitor, predict, and visualize electricity usage with the influence of renewable energy sources in Germany. It integrates machine learning (XGBoost & LSTM) models with a Streamlit dashboard, pulling live data from a PostgreSQL database.

Key Features
Data Integration: Fetches electricity load, weather, and renewable data directly from PostgreSQL.

Predictive Modeling:
XGBoost for fast, tree-based prediction.
LSTM for sequential deep learning forecasting.

Visualization Dashboard:
Actual vs. Predicted load comparisons
Renewable energy contribution
Correlation heatmaps
Feature importance (XGBoost)

Model Evaluation: Calculates MAE, RMSE, and R² for model performance comparison.

Automation: Model predictions are written back to the PostgreSQL database.


✅ Requirements
Python 3.8+
PostgreSQL
Libraries: pandas, numpy, streamlit, sqlalchemy, tensorflow, xgboost, seaborn, plotly, scikit-learn
