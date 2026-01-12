# AI Stock Price Predictor

An AI-powered financial market predictor supporting multiple global markets (NSE, BSE, NYSE, NASDAQ, etc.) and asset classes (Stocks, Crypto, Forex, Mutual Funds).

## Features
- **Multi-Market Support**: India, US, China, Europe, Japan, etc.
- **Asset Classes**: Stocks, Crypto, Forex, Mutual Funds.
- **AI Predictions**: Linear Regression, LSTM, etc. for future price trends.
- **Interactive Dashboard**: Real-time charts and indicators.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: Next.js (React)
- **ML**: TensorFlow, Scikit-learn, Pandas
- **Data**: yfinance

## Setup

### Backend
1. Navigate to `backend/`
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `uvicorn app.main:app --reload`

### Frontend
1. Navigate to `frontend/`
2. Install: `npm install`
3. Run: `npm run dev`
