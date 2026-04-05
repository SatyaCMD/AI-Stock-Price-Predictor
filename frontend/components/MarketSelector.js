"use client";
import { useState, useEffect } from 'react';
import axios from 'axios';
import { ChevronDown, Search, Globe, TrendingUp, DollarSign, ArrowRight } from 'lucide-react';

export default function MarketSelector({ onSelect }) {
    const [markets, setMarkets] = useState({});
    const [selectedRegion, setSelectedRegion] = useState('India');
    const [selectedMarket, setSelectedMarket] = useState('NSE');
    const [assetType, setAssetType] = useState('Stock');
    const [ticker, setTicker] = useState('');

    const FLAGS = {
        "India": "🇮🇳",
        "USA": "🇺🇸",
        "China": "🇨🇳",
        "Europe": "🇪🇺",
        "Japan": "🇯🇵",
        "Singapore": "🇸🇬",
        "South Korea": "🇰🇷",
        "Global": "🌍",
        "Crypto": "₿"
    };

    const [suggestions, setSuggestions] = useState([]);
    const [showSuggestions, setShowSuggestions] = useState(false);

    useEffect(() => {
        const fetchMarkets = async () => {
            let retries = 3;
            while (retries > 0) {
                try {
                    const res = await axios.get('/api/v1/markets');
                    setMarkets(res.data);
                    return; // Success, exit the loop
                } catch (error) {
                    retries -= 1;
                    if (retries === 0) {
                        console.error("Failed to fetch markets", error);
                        setMarkets({
                            "India": ["NSE", "BSE"],
                            "USA": ["NYSE", "NASDAQ"],
                            "South Korea": ["KOSPI", "KOSDAQ"],
                            "China": ["Shanghai", "Shenzhen"],
                            "Japan": ["Tokyo"],
                            "Singapore": ["SGX"],
                            "Global": ["All Markets"],
                            "Crypto": ["Binance"]
                        });
                    } else {
                        // Wait 1.5 seconds before retrying to allow backend to finish starting up
                        await new Promise(resolve => setTimeout(resolve, 1500));
                    }
                }
            }
        };
        fetchMarkets();
    }, []);

    useEffect(() => {
        const fetchSuggestions = async () => {
            if (ticker.length > 1) {
                try {
                    const searchRegion = assetType === 'Stock' ? selectedRegion : 'Global';
                    const res = await axios.get(`/api/v1/search?q=${ticker}&region=${searchRegion}&asset_type=${assetType}`);
                    setSuggestions(res.data);
                    setShowSuggestions(true);
                } catch (error) {
                    console.error("Failed to fetch suggestions", error);
                }
            } else {
                setSuggestions([]);
                setShowSuggestions(false);
            }
        };

        const timeoutId = setTimeout(fetchSuggestions, 300); // Debounce
        return () => clearTimeout(timeoutId);
    }, [ticker]);

    const handleSearch = (e) => {
        e.preventDefault();
        if (ticker) {
            let searchTicker = ticker.toUpperCase();

            // If the user entered a specific ticker with a suffix (e.g. TCS.NS, BTC-USD), use it directly.
            // Otherwise, apply the region/market specific suffix logic.
            const hasSuffix = searchTicker.includes('.') || (assetType === 'Crypto' && searchTicker.includes('-')) || (assetType === 'Forex' && searchTicker.includes('=X'));

            if (!hasSuffix) {
                // Auto-append suffixes based on selection if not present
                if (assetType === 'Crypto') {
                    searchTicker += '-USD';
                } else if (assetType === 'Forex') {
                    searchTicker += '=X';
                } else {
                    // Regional Suffixes
                    if (selectedRegion === 'India') {
                        if (selectedMarket === 'NSE') searchTicker += '.NSE';
                        if (selectedMarket === 'BSE') searchTicker += '.BSE';
                    } else if (selectedRegion === 'South Korea') {
                        if (selectedMarket === 'KOSPI') searchTicker += '.KS';
                        if (selectedMarket === 'KOSDAQ') searchTicker += '.KQ';
                    } else if (selectedRegion === 'China') {
                        if (selectedMarket === 'Shanghai') searchTicker += '.SS';
                        if (selectedMarket === 'Shenzhen') searchTicker += '.SZ';
                    } else if (selectedRegion === 'Japan') {
                        searchTicker += '.T';
                    } else if (selectedRegion === 'Singapore') {
                        searchTicker += '.SI';
                    } else if (selectedRegion === 'Global') {
                        // No suffix for Global search, user inputs exact ticker or we rely on default yfinance behavior (usually US)
                        // If user wants specific exchange, they should type it (e.g. RELIANCE.NS)
                    }
                    // USA typically doesn't need a suffix for major exchanges
                }
            }

            onSelect(searchTicker);
            setShowSuggestions(false);
        }
    };

    const handleSuggestionClick = (symbol) => {
        setTicker(symbol);
        onSelect(symbol);
        setShowSuggestions(false);
    };

    return (
        <div className="bg-white rounded-2xl mb-8 border border-gray-100 shadow-xl shadow-blue-500/5 overflow-visible z-40 relative">
            <div className="p-6 md:p-8">
                <div className="flex flex-col md:flex-row md:items-center justify-between mb-6">
                    <div className="flex items-center space-x-3 mb-4 md:mb-0">
                        <div className="bg-blue-50 p-2 rounded-lg">
                            <Search className="h-5 w-5 text-blue-600" />
                        </div>
                        <h2 className="text-xl font-semibold text-gray-900">Market Scanner</h2>
                    </div>
                    <div className="flex space-x-2 overflow-x-auto pb-2 md:pb-0">
                        {['Stocks', 'Crypto', 'Forex', 'Commodities', 'Funds'].map((type) => {
                            const typeMapping = {
                                'Stocks': 'Stock',
                                'Crypto': 'Crypto',
                                'Forex': 'Forex',
                                'Commodities': 'Commodity',
                                'Funds': 'Mutual Fund'
                            };
                            const targetType = typeMapping[type];
                            const isActive = assetType === targetType;

                            return (
                                <button
                                    key={type}
                                    onClick={() => setAssetType(targetType)}
                                    className={`px-4 py-1.5 rounded-full text-sm font-medium whitespace-nowrap transition-all ${isActive
                                        ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25'
                                        : 'bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-900 border border-gray-200'
                                        }`}
                                >
                                    {type}
                                </button>
                            );
                        })}
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-12 gap-4">

                    {/* Region Selector */}
                    {assetType === 'Stock' && (
                        <div className="md:col-span-3">
                            <label className="block text-xs font-medium text-gray-500 mb-1.5 ml-1">Region</label>
                            <div className="relative group">
                                <Globe className="absolute left-3 top-3.5 h-4 w-4 text-gray-400 group-focus-within:text-blue-600 transition-colors" />
                                <select
                                    className="w-full bg-white border border-gray-200 text-gray-900 rounded-xl pl-10 pr-8 py-3 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all hover:border-gray-300"
                                    value={selectedRegion}
                                    onChange={(e) => setSelectedRegion(e.target.value)}
                                >
                                    {Object.keys(markets).map((region) => (
                                        <option key={region} value={region}>
                                            {FLAGS[region] || "🌐"} {region}
                                        </option>
                                    ))}
                                </select>
                                <ChevronDown className="absolute right-3 top-3.5 h-4 w-4 text-gray-400 pointer-events-none" />
                            </div>
                        </div>
                    )}

                    {/* Market Selector */}
                    {assetType === 'Stock' && (
                        <div className="md:col-span-3">
                            <label className="block text-xs font-medium text-gray-500 mb-1.5 ml-1">Exchange</label>
                            <div className="relative group">
                                <TrendingUp className="absolute left-3 top-3.5 h-4 w-4 text-gray-400 group-focus-within:text-blue-600 transition-colors" />
                                <select
                                    className="w-full bg-white border border-gray-200 text-gray-900 rounded-xl pl-10 pr-8 py-3 appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all hover:border-gray-300"
                                    value={selectedMarket}
                                    onChange={(e) => setSelectedMarket(e.target.value)}
                                >
                                    {markets[selectedRegion]?.map((market) => (
                                        <option key={market} value={market}>{market}</option>
                                    ))}
                                </select>
                                <ChevronDown className="absolute right-3 top-3.5 h-4 w-4 text-gray-400 pointer-events-none" />
                            </div>
                        </div>
                    )}

                    {/* Search */}
                    <form onSubmit={handleSearch} className={`${assetType === 'Stock' ? 'md:col-span-6' : 'md:col-span-12'} flex flex-col md:flex-row gap-2 items-end relative`}>
                        <div className="w-full relative">
                            <label className="block text-xs font-medium text-gray-500 mb-1.5 ml-1">Search Symbol</label>
                            <div className="relative group">
                                <input
                                    type="text"
                                    value={ticker}
                                    onChange={(e) => setTicker(e.target.value)}
                                    onFocus={() => ticker.length > 1 && setShowSuggestions(true)}
                                    onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                                    placeholder={
                                        assetType === 'Crypto' ? "e.g., BTC, ETH, SOL" :
                                            assetType === 'Forex' ? "e.g., EURUSD, GBPUSD" :
                                                assetType === 'Commodity' ? "e.g., GC=F (Gold), CL=F (Oil)" :
                                                    assetType === 'Mutual Fund' ? "e.g., VFIAX, SWPPX" :
                                                        selectedRegion === 'India' ? "e.g., RELIANCE, TCS, INFY" :
                                                            "e.g., AAPL, MSFT, GOOGL"
                                    }
                                    className="w-full bg-white border border-gray-200 text-gray-900 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all placeholder-gray-400"
                                />
                            </div>

                            {/* Suggestions Dropdown */}
                            {showSuggestions && suggestions.length > 0 && (
                                <div className="absolute top-full left-0 w-full bg-white border border-gray-100 rounded-xl shadow-xl mt-2 max-h-60 overflow-y-auto z-50">
                                    {suggestions.map((s, i) => (
                                        <div
                                            key={i}
                                            onClick={() => handleSuggestionClick(s.symbol)}
                                            className="px-4 py-3 hover:bg-gray-50 cursor-pointer border-b border-gray-50 last:border-0 transition-colors"
                                        >
                                            <div className="flex justify-between items-center">
                                                <div>
                                                    <p className="font-semibold text-gray-900">{s.name}</p>
                                                    <p className="text-xs text-gray-500">{s.symbol}</p>
                                                </div>
                                                <div className="text-right">
                                                    <span className="text-xs font-medium bg-blue-50 text-blue-600 px-2 py-1 rounded-full">
                                                        {s.exchange}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        <button
                            type="submit"
                            className="w-full md:w-auto bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-all shadow-lg shadow-blue-500/20 active:scale-95 flex items-center justify-center whitespace-nowrap"
                        >
                            <span>Analyze</span>
                            <ArrowRight className="ml-2 h-4 w-4" />
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}
