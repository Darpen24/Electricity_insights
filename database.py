
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

db_config = {
    "drivername": "postgresql",
    "username": "postgres",
    "password": "darpen",  
    "host": "localhost",
    "port": 5433,
    "database": "Electricity_UsageMonitoring",
}

DATABASE_URL = URL.create(**db_config)

def fetch_data(query):
    """Fetch data from PostgreSQL and apply feature engineering."""
    print("Step 1: Creating a connection to the database...")
    try:
        engine = create_engine(DATABASE_URL)
        print("Connection established successfully!")

        print("Step 2: Running the query...")
        df = pd.read_sql(query, engine)
        print("Query executed successfully!")

        print("Step 3: Closing the connection...")
        engine.dispose()  
        print("Connection closed.")

        df = apply_feature_engineering(df)

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def apply_feature_engineering(df):
    """Apply feature engineering techniques on the dataset."""
    print("Applying feature engineering...")

    if 'timestamp_start' in df.columns:
        df['timestamp_start'] = pd.to_datetime(df['timestamp_start'])

        df['hour'] = df['timestamp_start'].dt.hour
        df['day'] = df['timestamp_start'].dt.day
        df['month'] = df['timestamp_start'].dt.month
        df['weekday'] = df['timestamp_start'].dt.weekday  
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    print("Feature engineering applied successfully!")
    return df

if __name__ == "__main__":
    print("Preparing to execute a test query...")
    query = "SELECT * FROM information_schema.tables LIMIT 5;"

    print("Fetching data...")
    result = fetch_data(query)

    if result is not None:
        print("Data fetched successfully! Here is the result:")
        print(result.head())
    else:
        print("Failed to fetch data.")
