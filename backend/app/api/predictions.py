from fastapi import APIRouter, HTTPException
from app.services import ml_service

router = APIRouter()

@router.get("/predict/{ticker}")
def predict_stock_price(ticker: str):
    """
    Get price predictions for a stock.
    """
    predictions = ml_service.get_predictions(ticker)
    if not predictions["linear_regression_prediction"] and not predictions["lstm_prediction"]:
        raise HTTPException(status_code=500, detail="Failed to generate predictions")
    return predictions
