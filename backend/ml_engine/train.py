import os
import pickle
import numpy as np
import pandas as pd
from app.services import data_service
from ml_engine import preprocessing, models

MODEL_DIR = "backend/ml_engine/saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_predict_linear_regression(ticker: str, period="2y"):
    """
    Train LR model and predict next day price.
    """
    # Fetch data
    data_dict = data_service.fetch_stock_data(ticker, period=period)
    if "error" in data_dict:
        return None
    
    df = pd.DataFrame(data_dict["history"])
    if df.empty:
        return None
        
    df = preprocessing.preprocess_data(df)
    if df.empty:
        return None
    
    # Prepare features for LR (using Days as feature)
    df['Days'] = (df.index - df.index[0]).days
    
    X = df[['Days']].values
    y = df['Close'].values
    
    model = models.train_linear_regression(X, y)
    
    # Predict next day
    next_day = X[-1][0] + 1
    prediction = model.predict([[next_day]])
    
    return prediction[0]

def train_predict_lstm(ticker: str, period="5y"):
    """
    Train LSTM model and predict next day price.
    """
    # Fetch data
    data_dict = data_service.fetch_stock_data(ticker, period=period)
    if "error" in data_dict:
        return None
        
    df = pd.DataFrame(data_dict["history"])
    if df.empty:
        return None
    
    # Preprocess
    x_train, y_train, scaler = preprocessing.prepare_lstm_data(df)
    
    # Build and train
    model = models.build_lstm_model((x_train.shape[1], 1))
    models.train_lstm_model(model, x_train, y_train, epochs=1, batch_size=1)
    
    # Predict next
    last_60_days = df['Close'].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    return pred_price[0][0]

def train_predict_logistic_regression(ticker: str, period="2y"):
    """
    Train Logistic Regression to predict if price will go UP (1) or DOWN (0).
    """
    # Fetch data
    data_dict = data_service.fetch_stock_data(ticker, period=period)
    if "error" in data_dict:
        return None
        
    df = pd.DataFrame(data_dict["history"])
    if df.empty:
        return None
        
    df = preprocessing.preprocess_data(df)
    if df.empty:
        return None
        
    # Feature Engineering for Classification
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Use simple features: Open, High, Low, Close, Volume
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[features].values[:-1] # Drop last row as it has no target
    y = df['Target'].values[:-1]
    
    if len(X) < 10: # Not enough data
        return None
        
    model = models.train_logistic_regression(X, y)
    
    # Predict for next day using latest data
    latest_data = df[features].values[-1].reshape(1, -1)
    prediction = model.predict(latest_data)
    
    return int(prediction[0]) # 1 for Up, 0 for Down
