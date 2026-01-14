from fastapi import APIRouter, HTTPException
from app.services import ml_service

router = APIRouter()

@router.get("/predict/{ticker}")
def predict_stock_price(ticker: str):
    """
    Get price predictions for a stock.
    """
    predictions = ml_service.get_predictions(ticker)
    if not predictions["linear_regression"] and not predictions["lstm"]:
        raise HTTPException(status_code=404, detail="Could not generate predictions (likely insufficient data)")
    return predictions
