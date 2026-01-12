from fastapi import APIRouter
from typing import List, Dict

router = APIRouter()

MARKETS = {
    "India": ["NSE", "BSE"],
    "USA": ["NYSE", "NASDAQ"],
    "China": ["Shanghai", "Shenzhen"],
    "Europe": ["London", "Euronext", "Deutsche Börse"],
    "Japan": ["Tokyo"],
    "Singapore": ["SGX"],
    "South Korea": ["KOSPI"],
}

ASSET_TYPES = [
    "Stock",
    "Crypto",
    "Forex",
    "Mutual Fund"
]

@router.get("/markets", response_model=Dict[str, List[str]])
def get_markets():
    """
    Get list of supported markets grouped by region.
    """
    return MARKETS

@router.get("/assets", response_model=List[str])
def get_asset_types():
    """
    Get list of supported asset types.
    """
    return ASSET_TYPES
