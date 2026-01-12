from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import data_service
from datetime import datetime, timedelta

router = APIRouter()

class InvestmentRequest(BaseModel):
    ticker: str
    amount: float
    duration_months: int = 12

@router.post("/calculate")
def calculate_investment(request: InvestmentRequest):
    """
    Calculate potential profit/loss based on historical data.
    """
    # Fetch historical data for the duration
    # We need data from (now - duration) to now
    # yfinance period format: "1y", "2y", "5y", "10y", "ytd", "max"
    # We'll approximate months to years or use specific dates if needed.
    # For simplicity, let's use "1y", "2y", "5y", "10y" mapping or just fetch "max" and slice.
    
    period = "1y"
    if request.duration_months > 12:
        period = "2y"
    if request.duration_months > 24:
        period = "5y"
    if request.duration_months > 60:
        period = "10y"
        
    data = data_service.fetch_stock_data(request.ticker, period=period)
    
    if "error" in data:
        raise HTTPException(status_code=404, detail=data["error"])
        
    history = data["history"]
    if not history:
        raise HTTPException(status_code=404, detail="No historical data found")
        
    # Get start and end price
    # Assuming history is sorted by Date
    # We need to find the price at (now - duration)
    
    # Simple logic: take the first available price in the fetched period 
    # (which roughly corresponds to the start of the period requested if we mapped correctly)
    # and the last price.
    
    start_price = history[0]["Close"]
    end_price = history[-1]["Close"]
    
    units = request.amount / start_price
    final_value = units * end_price
    profit = final_value - request.amount
    roi = (profit / request.amount) * 100
    
    return {
        "ticker": request.ticker,
        "initial_investment": request.amount,
        "duration_months": request.duration_months,
        "start_price": start_price,
        "end_price": end_price,
        "final_value": round(final_value, 2),
        "profit": round(profit, 2),
        "roi_percentage": round(roi, 2)
    }
