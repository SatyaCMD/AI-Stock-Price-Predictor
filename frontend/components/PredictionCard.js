import { TrendingUp, TrendingDown, Activity, Zap, Brain, ArrowUpRight, ArrowDownRight, Info } from 'lucide-react';
import { formatCurrency } from '../utils/currency';

export default function PredictionCard({ predictions, currency, currentPrice }) {
    if (!predictions) return null;

    const { linear_regression, lstm, logistic_regression } = predictions;

    // Helper to determine trend
    const getTrend = (predicted, current) => {
        if (!predicted || !current) return { label: 'Neutral', color: 'gray', Icon: Activity };
        const diff = predicted - current;
        const percentChange = (diff / current) * 100;

        if (percentChange > 0.5) return { label: 'Bullish', color: 'green', Icon: TrendingUp };
        if (percentChange < -0.5) return { label: 'Bearish', color: 'red', Icon: TrendingDown };
        return { label: 'Neutral', color: 'gray', Icon: Activity };
    };

    const lrTrend = getTrend(linear_regression?.prediction, currentPrice);
    const lstmTrend = getTrend(lstm?.prediction, currentPrice);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mt-8">
            {/* Linear Regression Card */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-xl shadow-blue-500/5 group hover:border-blue-500/30 transition-all duration-300 overflow-hidden">
                <div className="p-6 relative">
                    <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity transform group-hover:scale-110 duration-500">
                        <TrendingUp className="w-32 h-32 text-blue-600" />
                    </div>

                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center space-x-3">
                            <div className="p-2.5 bg-blue-50 rounded-xl border border-blue-100">
                                <Zap className="w-5 h-5 text-blue-600" />
                            </div>
                            <div>
                                <h3 className="text-base font-semibold text-black">Linear Regression</h3>
                                <p className="text-xs text-blue-600/80">Trend Analysis</p>
                            </div>
                        </div>
                        <div className={`px-2.5 py-1 rounded-full bg-${lrTrend.color}-50 border border-${lrTrend.color}-100 text-${lrTrend.color}-600 text-xs font-medium flex items-center`}>
                            {lrTrend.label === 'Bullish' ? <ArrowUpRight className="w-3 h-3 mr-1" /> :
                                lrTrend.label === 'Bearish' ? <ArrowDownRight className="w-3 h-3 mr-1" /> :
                                    <Activity className="w-3 h-3 mr-1" />}
                            {lrTrend.label}
                        </div>
                    </div>

                    <div className="space-y-1">
                        <p className="text-sm text-gray-700 font-medium">Predicted Price (Next Day)</p>
                        <div className="flex items-baseline space-x-2">
                            <span className="text-4xl md:text-5xl font-bold text-black tracking-tight">
                                {formatCurrency(linear_regression?.prediction, currency)}
                            </span>
                        </div>
                    </div>

                    <div className="mt-4 p-3 bg-blue-50/50 rounded-lg border border-blue-100/50">
                        <div className="flex items-start space-x-2">
                            <Info className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-blue-800 leading-relaxed">
                                {linear_regression?.analysis}
                            </p>
                        </div>
                    </div>

                    <div className="mt-6 pt-6 border-t border-gray-100 flex items-center justify-between">
                        <div className="text-xs text-gray-600 font-medium">Model Confidence</div>
                        <div className="flex items-center space-x-2">
                            <div className="w-24 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-1000"
                                    style={{ width: `${linear_regression?.confidence || 0}%` }}
                                ></div>
                            </div>
                            <span className="text-xs font-bold text-blue-600">
                                {linear_regression?.confidence ? linear_regression.confidence.toFixed(0) : 0}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* LSTM Card */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-xl shadow-purple-500/5 group hover:border-purple-500/30 transition-all duration-300 overflow-hidden">
                <div className="p-6 relative">
                    <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity transform group-hover:scale-110 duration-500">
                        <Brain className="w-32 h-32 text-purple-600" />
                    </div>

                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center space-x-3">
                            <div className="p-2.5 bg-purple-50 rounded-xl border border-purple-100">
                                <Brain className="w-5 h-5 text-purple-600" />
                            </div>
                            <div>
                                <h3 className="text-base font-semibold text-black">LSTM Neural Network</h3>
                                <p className="text-xs text-purple-600/80">Deep Learning</p>
                            </div>
                        </div>
                        <div className={`px-2.5 py-1 rounded-full bg-${lstmTrend.color}-50 border border-${lstmTrend.color}-100 text-${lstmTrend.color}-600 text-xs font-medium flex items-center`}>
                            {lstmTrend.label === 'Bullish' ? <ArrowUpRight className="w-3 h-3 mr-1" /> :
                                lstmTrend.label === 'Bearish' ? <ArrowDownRight className="w-3 h-3 mr-1" /> :
                                    <Activity className="w-3 h-3 mr-1" />}
                            {lstmTrend.label}
                        </div>
                    </div>

                    <div className="space-y-1">
                        <p className="text-sm text-gray-700 font-medium">Predicted Price (Next Day)</p>
                        <div className="flex items-baseline space-x-2">
                            <span className="text-4xl md:text-5xl font-bold text-black tracking-tight">
                                {formatCurrency(lstm?.prediction, currency)}
                            </span>
                        </div>
                    </div>

                    <div className="mt-4 p-3 bg-purple-50/50 rounded-lg border border-purple-100/50">
                        <div className="flex items-start space-x-2">
                            <Info className="w-4 h-4 text-purple-500 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-purple-800 leading-relaxed">
                                {lstm?.analysis}
                            </p>
                        </div>
                    </div>

                    <div className="mt-6 pt-6 border-t border-gray-100 flex items-center justify-between">
                        <div className="text-xs text-gray-600 font-medium">Model Confidence</div>
                        <div className="flex items-center space-x-2">
                            <div className="w-24 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-purple-600 to-purple-400 rounded-full transition-all duration-1000"
                                    style={{ width: `${lstm?.confidence || 0}%` }}
                                ></div>
                            </div>
                            <span className="text-xs font-bold text-purple-600">
                                {lstm?.confidence ? lstm.confidence.toFixed(0) : 0}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Logistic Regression Card */}
            <div className="bg-white rounded-2xl border border-gray-100 shadow-xl shadow-orange-500/5 group hover:border-orange-500/30 transition-all duration-300 overflow-hidden md:col-span-2 lg:col-span-1">
                <div className="p-6 relative">
                    <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity transform group-hover:scale-110 duration-500">
                        <Activity className="w-32 h-32 text-orange-600" />
                    </div>

                    <div className="flex items-center justify-between mb-6">
                        <div className="flex items-center space-x-3">
                            <div className="p-2.5 bg-orange-50 rounded-xl border border-orange-100">
                                <Activity className="w-5 h-5 text-orange-600" />
                            </div>
                            <div>
                                <h3 className="text-base font-semibold text-black">Logistic Regression</h3>
                                <p className="text-xs text-orange-600/80">Classification</p>
                            </div>
                        </div>
                    </div>

                    <div className="space-y-1">
                        <p className="text-sm text-gray-700 font-medium">Market Direction (Next Day)</p>
                        <div className="flex items-center space-x-3">
                            {logistic_regression?.prediction === 1 ? (
                                <>
                                    <span className="text-4xl md:text-5xl font-bold text-green-600 tracking-tight">Bullish</span>
                                    <TrendingUp className="w-8 h-8 text-green-500" />
                                </>
                            ) : logistic_regression?.prediction === 0 ? (
                                <>
                                    <span className="text-4xl md:text-5xl font-bold text-red-600 tracking-tight">Bearish</span>
                                    <TrendingDown className="w-8 h-8 text-red-500" />
                                </>
                            ) : (
                                <span className="text-4xl md:text-5xl font-bold text-gray-400 tracking-tight">Neutral</span>
                            )}
                        </div>
                    </div>

                    <div className="mt-4 p-3 bg-orange-50/50 rounded-lg border border-orange-100/50">
                        <div className="flex items-start space-x-2">
                            <Info className="w-4 h-4 text-orange-500 mt-0.5 flex-shrink-0" />
                            <p className="text-xs text-orange-800 leading-relaxed">
                                {logistic_regression?.analysis}
                            </p>
                        </div>
                    </div>

                    <div className="mt-6 pt-6 border-t border-gray-100 flex items-center justify-between">
                        <div className="text-xs text-gray-600 font-medium">Model Confidence</div>
                        <div className="flex items-center space-x-2">
                            <div className="w-24 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-orange-600 to-orange-400 rounded-full transition-all duration-1000"
                                    style={{ width: `${logistic_regression?.confidence || 0}%` }}
                                ></div>
                            </div>
                            <span className="text-xs font-bold text-orange-600">
                                {logistic_regression?.confidence ? logistic_regression.confidence.toFixed(0) : 0}%
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
