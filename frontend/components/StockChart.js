"use client";
import { AdvancedRealTimeChart } from "react-ts-tradingview-widgets";
import { useEffect, useState, memo } from 'react';

const StockChart = memo(function StockChart({ ticker }) {
    const [tvTicker, setTvTicker] = useState("AAPL");

    useEffect(() => {
        if (ticker) {
            // TradingView expects format like 'BSE:RELIANCE' or 'NASDAQ:AAPL'
            // We'll strip .NS/.BO and prepend exchange based on suffix if present
            let formatted = ticker;
            if (ticker.endsWith('.NS') || ticker.endsWith('.NSE') || ticker.endsWith('.BO') || ticker.endsWith('.BSE')) {
                // NSE generally restricts embedding in free TradingView widgets. Default to BSE.
                formatted = `BSE:${ticker.replace(/\.NSE?|\.BSE?$/, '')}`;
            } else if (ticker.endsWith('.T')) {
                formatted = `TSE:${ticker.replace('.T', '')}`;
            } else if (ticker.endsWith('.KS')) {
                formatted = `KRX:${ticker.replace('.KS', '')}`;
            } else if (ticker.endsWith('.SS')) {
                formatted = `SSE:${ticker.replace('.SS', '')}`;
            } else if (ticker.endsWith('.HK')) {
                formatted = `HKEX:${ticker.replace('.HK', '')}`;
            } else if (ticker.endsWith('.SZ')) {
                formatted = `SZSE:${ticker.replace('.SZ', '')}`;
            } else if (ticker.startsWith('^')) {
                // Indices
                if (ticker === '^NSEI' || ticker === '^BSESN') formatted = 'BSE:SENSEX';
                else if (ticker === '^NDX') formatted = 'NDX';
                else if (ticker === '^GSPC') formatted = 'SPX';
                else if (ticker === '^DJI') formatted = 'DJI';
                else if (ticker === '^N225') formatted = 'NKY';
                else if (ticker === '^FTSE') formatted = 'UKX';
                else formatted = ticker.replace('^', '');
            } else if (ticker.includes('-USD')) {
                // Crypto from Yahoo (e.g. BTC-USD -> BINANCE:BTCUSDT)
                formatted = `BINANCE:${ticker.replace('-USD', 'USDT')}`;
            } else if (ticker.endsWith('=X')) {
                // Forex from Yahoo (e.g. EURUSD=X -> FX:EURUSD)
                formatted = `FX_IDC:${ticker.replace('=X', '')}`;
            } else if (ticker.endsWith('=F')) {
                // Commodities/Futures from Yahoo
                const base = ticker.replace('=F', '');
                if (['GC', 'SI', 'HG'].includes(base)) formatted = `COMEX:${base}1!`;
                else if (['CL', 'NG'].includes(base)) formatted = `NYMEX:${base}1!`;
                else formatted = base;
            }
            // If it's pure alphabet like AAPL or TSLA, we let TV auto-resolve it.
            setTvTicker(formatted);
        }
    }, [ticker]);

    return (
        <div className="glass-card w-full h-[700px] rounded-2xl border border-gray-200 bg-white shadow-sm overflow-hidden relative">
            <AdvancedRealTimeChart
                key={tvTicker}
                symbol={tvTicker}
                theme="light"
                autosize
                interval="D"
                timezone={Intl.DateTimeFormat().resolvedOptions().timeZone || "Etc/UTC"}
                allow_symbol_change={true}
                hide_side_toolbar={false}
                hide_top_toolbar={false}
                save_image={false}
                details={false}
                hotlist={false}
                calendar={false}
                studies={[
                    "STD;MACD",
                    "STD;RSI"
                ]}
            />
        </div>
    );
});

export default StockChart;
