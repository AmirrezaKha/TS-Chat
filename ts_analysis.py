%pip install pandas requests pyspark delta-spark plotly xgboost scikit-learn numpy


import requests
import pandas as pd
import time
from pyspark.sql import SparkSession
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np


class StockForecasting:
    def __init__(self, api_key):
        self.api_key = api_key
        self.spark = SparkSession.builder.appName("StockForecasting").getOrCreate()
        self.delta_path = "/mnt/delta/stock_data_test"

    def fetch_stock_data(self, symbol):
        """
        Fetch stock data using Alpha Vantage API.
        """
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}&outputsize=full'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df = df.rename(columns={
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'
                })
                df.index = pd.to_datetime(df.index)
                df.reset_index(inplace=True)
                df = df.rename(columns={'index': 'time'})
                df['symbol'] = symbol
                return df
            else:
                print("Error: Data not available or request limit exceeded.")
                return None
        else:
            print(f"Error: Failed to fetch data. Status code: {response.status_code}")
            return None

    def save_to_delta(self, df):
        """
        Save Pandas DataFrame to Delta Lake.
        """
        stock_data_spark = self.spark.createDataFrame(df)
        stock_data_spark.write.format("delta").mode("overwrite").save(self.delta_path)
        print(f"Data saved to Delta Lake at: {self.delta_path}")

    def plot_stock_data(self, symbol):
        """
        Load data from Delta Lake and plot stock price.
        """
        df = self.spark.read.format("delta").load(self.delta_path).filter(f"symbol = '{symbol}'").toPandas()
        df['time'] = pd.to_datetime(df['time'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).sort_values(by='time')

        fig = px.line(df, x='time', y='close', title=f'{symbol} Stock Price Over Time', labels={'time': 'Time', 'close': 'Close Price'})
        fig.update_xaxes(tickformat='%Y-%m-%d', tickangle=45)
        fig.update_yaxes(tickformat=".2f")
        fig.update_layout(template="plotly_dark", autosize=True)
        fig.show()

    def train_xgboost_model(self, symbol, lags=5):
        """
        Train an XGBoost model for time series forecasting.
        """
        df = self.spark.read.format("delta").load(self.delta_path).filter(f"symbol = '{symbol}'").toPandas()
        df['time'] = pd.to_datetime(df['time'])
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).sort_values(by='time')

        # Create lag features
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = df['close'].shift(lag)
        df = df.dropna()

        X = df[[f'lag_{lag}' for lag in range(1, lags + 1)]]
        y = df['close']
        tscv = TimeSeriesSplit(n_splits=5)

        # Hyperparameter tuning
        model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
        }

        grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_root_mean_squared_error', verbose=1)
        grid_search.fit(X, y)
        best_model = xgb.XGBRegressor(**grid_search.best_params_)
        best_model.fit(X, y)

        # Evaluate model
        test_size = int(0.2 * len(df))
        X_test, y_test = X[-test_size:], y[-test_size:]
        y_pred_test = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        print(f"Test RMSE: {test_rmse}")

        return best_model, df, X_test, y_test, y_pred_test

    def plot_model_results(self, df, X_test, y_test, y_pred_test):
        """
        Plot actual vs predicted results.
        """
        test_results = pd.DataFrame({
            'Time': df['time'].iloc[-len(y_test):],
            'Actual': y_test,
            'Predicted': y_pred_test
        })

        fig = px.line(title='Actual vs Predicted Stock Prices')
        fig.add_scatter(x=test_results['Time'], y=test_results['Actual'], mode='lines', name='Actual', line=dict(color='blue'))
        fig.add_scatter(x=test_results['Time'], y=test_results['Predicted'], mode='lines', name='Predicted', line=dict(color='orange', dash='dot'))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Close Price",
            template="plotly_dark",
        )
        fig.show()
