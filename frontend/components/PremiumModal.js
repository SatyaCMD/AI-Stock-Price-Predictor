import React from 'react';
import { X, CheckCircle2, Shield, Calendar, User, Star, Clock } from 'lucide-react';

export default function PremiumModal({ isOpen, onClose, userProfile }) {
    if (!isOpen) return null;

    const isActive = userProfile?.isSubscribed;
    
    // Formatting dates fallback
    const fallbackStart = userProfile?.joinDate ? new Date(userProfile.joinDate) : new Date();
    const fallbackEnd = new Date(fallbackStart.getTime() + 30 * 24 * 60 * 60 * 1000);

    const startObj = userProfile?.subscriptionStartDate ? new Date(userProfile.subscriptionStartDate) : fallbackStart;
    const endObj = userProfile?.subscriptionNextDate ? new Date(userProfile.subscriptionNextDate) : fallbackEnd;

    const startDate = startObj.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
    const endDate = endObj.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-gray-900/40 backdrop-blur-sm p-4">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden animate-in zoom-in-95 duration-200 border border-yellow-100">
                
                {/* Header Phase: Vibrant and Eye-catching */}
                <div className={`relative px-5 py-5 text-center ${isActive ? 'bg-gradient-to-br from-yellow-400 via-amber-400 to-orange-400' : 'bg-gradient-to-br from-slate-800 to-slate-900'}`}>
                    <button 
                        onClick={onClose}
                        className={`absolute top-4 right-4 p-1.5 rounded-full ${isActive ? 'bg-yellow-500/30 text-yellow-900 hover:bg-yellow-500/50' : 'bg-white/10 text-gray-300 hover:bg-white/20'} transition-colors`}
                    >
                        <X className="w-5 h-5" />
                    </button>
                    
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-full shadow-inner mb-3 ${isActive ? 'bg-white text-amber-500 shadow-yellow-600/20' : 'bg-slate-700 text-yellow-400 shadow-black/20'}`}>
                        <Star className={`w-6 h-6 ${isActive ? 'fill-amber-400' : ''}`} />
                    </div>
                    
                    <h2 className={`text-2xl font-extrabold tracking-tight ${isActive ? 'text-yellow-950' : 'text-white'}`}>
                        TradeMind AI Premium
                    </h2>
                    
                    {isActive ? (
                        <div className="mt-3 inline-flex items-center px-4 py-1.5 rounded-full bg-white/30 border border-white/40 text-yellow-950 font-bold text-sm shadow-sm">
                            <Shield className="w-4 h-4 mr-2" />
                            Active Subscription
                        </div>
                    ) : (
                        <p className="mt-2 text-slate-300 text-sm">
                            Unlock the full power of autonomous AI trading.
                        </p>
                    )}
                </div>

                <div className="p-5">
                    {/* User Subscription Details (If Active) */}
                    {isActive && (
                        <div className="mb-5 p-4 bg-gradient-to-br from-gray-50 to-amber-50 border border-amber-100/50 rounded-xl space-y-3">
                            <h3 className="text-sm font-bold text-gray-500 uppercase tracking-widest border-b border-gray-200 pb-2">Account Details</h3>
                            
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-xs text-gray-500 flex items-center mb-1"><User className="w-3 h-3 mr-1" /> Subscriber</p>
                                    <p className="font-bold text-gray-900">{userProfile.name}</p>
                                    <p className="text-xs text-gray-500 font-mono mt-0.5">{userProfile.userId}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-gray-500 flex items-center justify-end mb-1"><Star className="w-3 h-3 mr-1" /> Plan Tier</p>
                                    <p className="font-bold text-amber-600 capitalize">{userProfile.subscriptionPlan || 'Monthly'} Premium</p>
                                </div>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-4 pt-3 border-t border-amber-100/30">
                                <div>
                                    <p className="text-xs text-gray-500 flex items-center mb-1"><Calendar className="w-3 h-3 mr-1" /> Subscribed On</p>
                                    <p className="font-semibold text-gray-800 text-sm">{startDate}</p>
                                </div>
                                <div className="text-right">
                                    <p className="text-xs text-gray-500 flex items-center justify-end mb-1"><Clock className="w-3 h-3 mr-1" /> Next Cycle</p>
                                    <p className="font-semibold text-gray-800 text-sm">{endDate}</p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Features List */}
                    <div>
                        <h3 className="text-sm font-bold text-gray-500 uppercase tracking-widest mb-3">
                            {isActive ? "Your Premium Benefits" : "Premium Facilities Included"}
                        </h3>
                        
                        <ul className="space-y-3">
                            <li className="flex items-start">
                                <CheckCircle2 className="w-5 h-5 text-green-500 mr-3 shrink-0" />
                                <div>
                                    <p className="text-sm font-bold text-gray-900">Advanced AI Market Scanners</p>
                                    <p className="text-xs text-gray-500 mt-0.5">Real-time global anomaly detection across 10+ markets including Commodities and Crypto.</p>
                                </div>
                            </li>
                            <li className="flex items-start">
                                <CheckCircle2 className="w-5 h-5 text-green-500 mr-3 shrink-0" />
                                <div>
                                    <p className="text-sm font-bold text-gray-900">Custom Branded PDF Reports</p>
                                    <p className="text-xs text-gray-500 mt-0.5">Generate and download unlimited professional transaction history reports with dynamic logos.</p>
                                </div>
                            </li>
                            <li className="flex items-start">
                                <CheckCircle2 className="w-5 h-5 text-green-500 mr-3 shrink-0" />
                                <div>
                                    <p className="text-sm font-bold text-gray-900">Automated AI Trading Signals</p>
                                    <p className="text-xs text-gray-500 mt-0.5">Get instant predictions based on CNNs, LSTMs, and XGBoost models backed by robust ML analytics.</p>
                                </div>
                            </li>
                            <li className="flex items-start">
                                <CheckCircle2 className="w-5 h-5 text-green-500 mr-3 shrink-0" />
                                <div>
                                    <p className="text-sm font-bold text-gray-900">Zero-Delay Order Execution</p>
                                    <p className="text-xs text-gray-500 mt-0.5">Priority routing for pending orders allowing immediate action upon market opening.</p>
                                </div>
                            </li>
                        </ul>
                    </div>
                </div>
                
                <div className="p-3 bg-gray-50 border-t border-gray-100 flex justify-end rounded-b-2xl">
                    <button 
                        onClick={onClose}
                        className="px-6 py-2.5 bg-gray-900 text-white text-sm font-bold rounded-xl hover:bg-gray-800 transition-colors shadow-sm"
                    >
                        {isActive ? 'Close' : 'Maybe Later'}
                    </button>
                </div>
            </div>
        </div>
    );
}
