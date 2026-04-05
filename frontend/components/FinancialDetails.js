import { formatCurrency } from '../utils/currency';

export default function FinancialDetails({ info }) {
    if (!info) return null;

    const formatNumber = (num) => {
        if (!num) return 'N/A';
        if (num >= 1.0e+12) return (num / 1.0e+12).toFixed(2) + "T";
        if (num >= 1.0e+9) return (num / 1.0e+9).toFixed(2) + "B";
        if (num >= 1.0e+6) return (num / 1.0e+6).toFixed(2) + "M";
        return num.toLocaleString();
    };

    const details = [
        { label: "Market Cap", value: formatNumber(info.marketCap), prefix: info.currency === 'USD' ? '$' : '' }, // Simplified prefix
        { label: "P/E Ratio", value: info.peRatio?.toFixed(2) },
        { label: "Forward P/E", value: info.forwardPE?.toFixed(2) },
        { label: "EPS (TTM)", value: info.eps?.toFixed(2) },
        { label: "Beta", value: info.beta?.toFixed(2) },
        { label: "52W High", value: formatCurrency(info.fiftyTwoWeekHigh, info.currency) },
        { label: "52W Low", value: formatCurrency(info.fiftyTwoWeekLow, info.currency) },
        { label: "Div Yield", value: info.dividendYield ? (info.dividendYield * 100).toFixed(2) + '%' : 'N/A' },
        { label: "Avg Volume", value: formatNumber(info.averageVolume) },
        { label: "Profit Margin", value: info.profitMargins ? (info.profitMargins * 100).toFixed(2) + '%' : 'N/A' },
    ];

    return (
        <div className="bg-white rounded-2xl border border-gray-100 shadow-xl shadow-blue-500/5 p-6 mt-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Financial Overview</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {details.map((item, index) => (
                    <div key={index} className="p-3 bg-gray-50 rounded-xl border border-gray-100">
                        <p className="text-xs text-gray-500 font-medium mb-1">{item.label}</p>
                        <p className="text-sm font-bold text-gray-900">
                            {item.prefix}{item.value || 'N/A'}
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
}
