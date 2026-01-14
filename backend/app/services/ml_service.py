from ml_engine import train

def get_predictions(ticker: str):
    """
    Get predictions from all models.
    """
    lr_result = train.train_predict_linear_regression(ticker)
    lstm_result = train.train_predict_lstm(ticker)
    log_reg_result = train.train_predict_logistic_regression(ticker)
    
    return {
        "ticker": ticker,
        "linear_regression": lr_result,
        "lstm": lstm_result,
        "logistic_regression": log_reg_result
    }
