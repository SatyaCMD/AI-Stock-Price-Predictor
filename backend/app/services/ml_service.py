from ml_engine import train

def get_predictions(ticker: str):
    """
    Get predictions from all models.
    """
    lr_prediction = train.train_predict_linear_regression(ticker)
    lstm_prediction = train.train_predict_lstm(ticker)
    log_reg_prediction = train.train_predict_logistic_regression(ticker)
    
    return {
        "ticker": ticker,
        "linear_regression_prediction": float(lr_prediction) if lr_prediction else None,
        "lstm_prediction": float(lstm_prediction) if lstm_prediction else None,
        "logistic_regression_prediction": log_reg_prediction # 0 or 1, or None
    }
