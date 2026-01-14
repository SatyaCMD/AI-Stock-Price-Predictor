import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from app.services import data_service
from ml_engine import preprocessing, models

MODEL_DIR = "backend/ml_engine/saved_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def train_predict_linear_regression(ticker: str, period="2y"):
    """
    Train LR model and predict next day price with analysis.
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
    
    # Prepare features for LR (using Days and Indicators)
    df['Days'] = (df.index - df.index[0]).days
    
    # Features: Days, MA_50, RSI
    feature_cols = ['Days', 'MA_50', 'RSI']
    # Ensure we have these columns
    for col in feature_cols:
        if col not in df.columns:
            return None

    X = df[feature_cols].values
    y = df['Close'].values

    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = models.train_linear_regression(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    confidence = max(0, min(98, r2 * 100)) # Clip between 0 and 98
    
    # Retrain on full data for final prediction
    model_full = models.train_linear_regression(X, y)

    # Predict next day
    # We need to estimate next day's features. 
    # Days = last + 1
    # MA_50 = approx last MA_50 (simplification)
    # RSI = approx last RSI
    last_row = df.iloc[-1]
    next_day_features = [
        last_row['Days'] + 1,
        last_row['MA_50'],
        last_row['RSI']
    ]
    
    prediction = model_full.predict([next_day_features])[0]
    
    # Analysis
    trend = "UP" if prediction > last_row['Close'] else "DOWN"
    rsi_val = last_row['RSI']
    rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
    
    analysis = (
        f"The Linear Regression model predicts a {trend} trend with {confidence:.1f}% confidence. "
        f"It incorporates the 50-day Moving Average and RSI ({rsi_val:.1f}, {rsi_status}) to adjust for momentum. "
        f"While the long-term trend is the primary driver, the RSI suggests current market sentiment is {rsi_status.lower()}."
    )

    return {
        "prediction": float(prediction),
        "confidence": float(confidence),
        "analysis": analysis
    }

def train_predict_lstm(ticker: str, period="5y"):
    """
    Train LSTM model and predict next day price with analysis.
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
    # Increased epochs for better accuracy
    history = models.train_lstm_model(model, x_train, y_train, epochs=25, batch_size=32)
    
    # Calculate "confidence" based on final loss
    final_loss = history.history['loss'][-1]
    confidence = max(0, min(95, (1 - final_loss * 10) * 100)) # Heuristic scaling
    
    # Predict next
    last_60_days = df['Close'].values[-60:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    prediction = pred_price[0][0]
    
    # Analysis
    current_price = df['Close'].iloc[-1]
    change = ((prediction - current_price) / current_price) * 100
    action = "BUY" if change > 1 else "SELL" if change < -1 else "HOLD"
    
    analysis = (
        f"The LSTM Neural Network identifies a potential {change:.2f}% move ({action} signal) with {confidence:.1f}% confidence. "
        f"By analyzing complex sequential patterns over the last 60 days, the model detects non-linear trends missed by traditional indicators. "
        f"The deep learning architecture suggests the price is gravitating towards {prediction:.2f}."
    )
    
    return {
        "prediction": float(prediction),
        "confidence": float(confidence),
        "analysis": analysis
    }

def train_predict_logistic_regression(ticker: str, period="2y"):
    """
    Train Logistic Regression to predict if price will go UP (1) or DOWN (0) with analysis.
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
    
    # Enhanced Features: RSI, MACD, Signal, MA_50
    features = ['Open', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 'MA_50']
    # Ensure columns exist
    for col in features:
        if col not in df.columns:
            return None

    X = df[features].values[:-1] # Drop last row as it has no target
    y = df['Target'].values[:-1]
    
    if len(X) < 10: # Not enough data
        return None
        
    # Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = models.train_logistic_regression(X_train, y_train)
    
    # Evaluate
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    confidence = accuracy * 100
    
    # Retrain on full data
    model_full = models.train_logistic_regression(X, y)
    
    # Predict for next day using latest data
    latest_data = df[features].values[-1].reshape(1, -1)
    prediction = model_full.predict(latest_data)[0]
    
    direction = "UP" if prediction == 1 else "DOWN"
    
    # Analysis based on indicators
    last_row = df.iloc[-1]
    macd_signal = "Bullish" if last_row['MACD'] > last_row['Signal_Line'] else "Bearish"
    
    analysis = (
        f"The Logistic Regression classifier predicts an {direction} movement with {confidence:.1f}% historical accuracy. "
        f"Key drivers include the MACD ({macd_signal} crossover) and RSI ({last_row['RSI']:.1f}). "
        f"This model combines volume analysis with momentum indicators to determine the most probable market direction."
    )
    
    return {
        "prediction": int(prediction),
        "confidence": float(confidence),
        "analysis": analysis
    }
