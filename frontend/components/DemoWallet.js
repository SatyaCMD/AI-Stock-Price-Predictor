"use client";
import React, { useState, useEffect } from 'react';
import { Wallet, Plus, Minus, Download, X, History, Globe, Loader2, CheckCircle2 } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { getLogoDataUrl, addInteractiveLogoToPage } from '../utils/pdfHelper';

export default function DemoWallet({ isOpen, onClose }) {
    const [activeMarket, setActiveMarket] = useState('US');
    const [balances, setBalances] = useState({ US: 0, IN: 0, CN: 0, EU: 0, JP: 0, SG: 0, KR: 0, CRYPTO: 0, FOREX: 0, COMMODITIES: 0 });
    const [amount, setAmount] = useState('');
    const [transactions, setTransactions] = useState({ US: [], IN: [], CN: [], EU: [], JP: [], SG: [], KR: [], CRYPTO: [], FOREX: [], COMMODITIES: [] });
    const [paymentStatus, setPaymentStatus] = useState('idle'); // 'idle', 'processing', 'success'
    const [pendingTx, setPendingTx] = useState(null);
    const [error, setError] = useState(null);

    const markets = [
        { id: 'US', name: 'US Market', currency: '$', symbol: 'USD' },
        { id: 'IN', name: 'Indian Market', currency: '₹', symbol: 'INR' },
        { id: 'CN', name: 'Chinese Market', currency: '¥', symbol: 'CNY' },
        { id: 'EU', name: 'European Market', currency: '€', symbol: 'EUR' },
        { id: 'JP', name: 'Japanese Market', currency: '¥', symbol: 'JPY' },
        { id: 'SG', name: 'Singapore Market', currency: 'S$', symbol: 'SGD' },
        { id: 'KR', name: 'South Korean Market', currency: '₩', symbol: 'KRW' },
        { id: 'CRYPTO', name: 'Crypto Market', currency: '₮', symbol: 'USDT' },
        { id: 'FOREX', name: 'Forex Market', currency: '$', symbol: 'USD' },
        { id: 'COMMODITIES', name: 'Commodities Market', currency: '$', symbol: 'USD' }
    ];

    const currentMarket = markets.find(m => m.id === activeMarket);
    const balance = balances[activeMarket] || 0;
    const marketTransactions = transactions[activeMarket] || [];

    useEffect(() => {
        if (typeof window !== 'undefined') {
            const savedBalances = localStorage.getItem('demoWalletBalances');
            const savedTransactions = localStorage.getItem('demoWalletTransactionsAll');

            if (savedBalances) {
                const parsed = JSON.parse(savedBalances);
                if (parsed.India !== undefined) { parsed.IN = parsed.India; delete parsed.India; }
                if (parsed.Crypto !== undefined) { parsed.CRYPTO = parsed.Crypto; delete parsed.Crypto; }
                setBalances(prev => ({ ...prev, ...parsed }));
            }
            if (savedTransactions) {
                const parsed = JSON.parse(savedTransactions);
                if (parsed.India !== undefined) { parsed.IN = parsed.India; delete parsed.India; }
                if (parsed.Crypto !== undefined) { parsed.CRYPTO = parsed.Crypto; delete parsed.Crypto; }
                setTransactions(prev => ({ ...prev, ...parsed }));
            }
        }
    }, [isOpen]);

    const saveState = (newBalances, newTransactions) => {
        setBalances(newBalances);
        setTransactions(newTransactions);
        localStorage.setItem('demoWalletBalances', JSON.stringify(newBalances));
        localStorage.setItem('demoWalletTransactionsAll', JSON.stringify(newTransactions));
    };

    const handleTransactionClick = (type) => {
        setError(null);
        const val = parseFloat(amount);

        if (isNaN(val) || val <= 0) {
            setError("Please enter a valid amount greater than 0.");
            return;
        }

        if (type === 'withdraw' && val > balance) {
            setError(`Insufficient funds! Your balance is ${currentMarket.currency}${balance.toFixed(2)}.`);
            return;
        }

        setPendingTx({ type, amount: val });
        setPaymentStatus('processing');

        // Simulate payment gateway delay
        setTimeout(() => {
            setPaymentStatus('success');
            setTimeout(() => {
                executeTransaction(type, val);
            }, 1200);
        }, 1500);
    };

    const executeTransaction = (type, val) => {
        const newBalance = type === 'add' ? balance + val : balance - val;

        const newBalances = { ...balances, [activeMarket]: newBalance };

        const newTransaction = {
            id: Date.now().toString(),
            date: new Date().toLocaleString(),
            type,
            amount: val,
            balanceAfter: newBalance,
            currency: currentMarket.symbol
        };

        const newTransactions = {
            ...transactions,
            [activeMarket]: [newTransaction, ...marketTransactions]
        };

        saveState(newBalances, newTransactions);
        setAmount('');
        setPaymentStatus('idle');
        setPendingTx(null);
    };

    const downloadPDF = async () => {
        const doc = new jsPDF();
        
        // Add interactive logo
        const pageWidth = doc.internal.pageSize.getWidth();
        const logoDataUrl = await getLogoDataUrl();
        addInteractiveLogoToPage(doc, pageWidth, logoDataUrl);

        doc.setFontSize(22);
        doc.setFont(undefined, 'bold');
        doc.setTextColor(30, 58, 138);
        doc.text("Wallet Transactions", 14, 32);

        // Divider Line
        doc.setDrawColor(226, 232, 240);
        doc.setLineWidth(0.5);
        doc.line(14, 38, pageWidth - 14, 38);

        // Add User Info
        const savedProfile = localStorage.getItem('userProfile');
        let userName = 'Guest';
        let userId = 'N/A';
        if (savedProfile) {
            const profile = JSON.parse(savedProfile);
            if (profile.name) userName = profile.name;
            if (profile.userId) userId = profile.userId;
        }

        doc.setFontSize(10);
        doc.setFont(undefined, 'normal');
        doc.setTextColor(100);
        doc.text(`Investor: ${userName} (ID: ${userId})`, 14, 46);
        doc.text(`Generated on: ${new Date().toLocaleString()}`, 14, 51);

        // Metadata Box
        doc.setFillColor(248, 250, 252);
        doc.setDrawColor(226, 232, 240);
        doc.roundedRect(14, 56, pageWidth - 28, 18, 3, 3, 'FD');

        doc.setFontSize(11);
        doc.setFont(undefined, 'bold');
        doc.setTextColor(15, 23, 42);
        doc.text(`Market: ${currentMarket.name}   |   Current Balance: ${currentMarket.currency}${balance.toFixed(2)} ${currentMarket.symbol}`, 20, 67);

        const tableColumn = ["Date", "Type", "Amount", "Balance After"];
        const tableRows = [];

        marketTransactions.forEach(t => {
            const transactionData = [
                t.date,
                t.type === 'add' ? 'Deposit' : 'Withdrawal',
                `${t.type === 'add' ? '+' : '-'}${currentMarket.currency}${t.amount.toFixed(2)}`,
                `${currentMarket.currency}${t.balanceAfter.toFixed(2)}`
            ];
            tableRows.push(transactionData);
        });

        autoTable(doc, {
            head: [tableColumn],
            body: tableRows,
            startY: 82,
            theme: 'striped',
            headStyles: { fillColor: [37, 99, 235], textColor: 255, fontStyle: 'bold' },
            bodyStyles: { textColor: 50 },
            alternateRowStyles: { fillColor: [248, 250, 252] },
            styles: { fontSize: 10, cellPadding: 5 }
        });

        doc.save(`demo_${currentMarket.id}_transactions_${Date.now()}.pdf`);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/40 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative w-full max-w-lg bg-white rounded-2xl shadow-2xl p-6 transform transition-all flex flex-col max-h-[90vh] overflow-hidden">

                {/* Payment Gateway Overlay */}
                {paymentStatus !== 'idle' && (
                    <div className="absolute inset-0 z-50 bg-white/95 backdrop-blur-md flex flex-col items-center justify-center animate-in fade-in duration-300">
                        {paymentStatus === 'processing' ? (
                            <div className="flex flex-col items-center space-y-4">
                                <Loader2 className="w-16 h-16 text-blue-600 animate-spin" />
                                <h3 className="text-xl font-bold text-gray-900">Processing Transaction...</h3>
                                <p className="text-gray-500 text-center px-6">
                                    Securely {pendingTx?.type === 'add' ? 'depositing' : 'withdrawing'} <span className="font-bold text-gray-900">{currentMarket.currency}{pendingTx?.amount.toFixed(2)} {currentMarket.symbol}</span>
                                </p>
                            </div>
                        ) : (
                            <div className="flex flex-col items-center space-y-4">
                                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                                    <CheckCircle2 className="w-10 h-10 text-green-600 animate-in zoom-in duration-300" />
                                </div>
                                <h3 className="text-xl font-bold text-gray-900">Transaction Successful!</h3>
                                <p className="text-green-600 font-medium text-lg">
                                    {pendingTx?.type === 'add' ? '+' : '-'}{currentMarket.currency}{pendingTx?.amount.toFixed(2)} {currentMarket.symbol}
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>

                {/* Header */}
                <div className="flex items-center space-x-3 mb-6">
                    <div className="p-3 bg-blue-100 rounded-xl">
                        <Wallet className="w-6 h-6 text-blue-600" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-gray-900">Demo Wallet</h2>
                        <p className="text-gray-500 text-sm">Manage your virtual funds</p>
                    </div>
                </div>

                {/* Market Selector */}
                <div className="flex items-center space-x-2 mb-4 overflow-hidden">
                    <Globe className="w-5 h-5 text-gray-400 flex-shrink-0" />
                    <div className="flex bg-gray-100 p-1 rounded-lg w-full overflow-x-auto scrollbar-hide snap-x">
                        {markets.map(m => (
                            <button
                                key={m.id}
                                onClick={() => setActiveMarket(m.id)}
                                className={`flex-none px-4 py-1.5 text-sm font-medium rounded-md transition-all whitespace-nowrap snap-start ${activeMarket === m.id
                                    ? 'bg-white text-gray-900 shadow-sm'
                                    : 'text-gray-500 hover:text-gray-700'
                                    }`}
                            >
                                {m.name}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Balance Card */}
                <div className="bg-gradient-to-br from-blue-600 to-indigo-700 rounded-2xl p-6 text-white mb-6 shadow-lg relative overflow-hidden transition-all">
                    <div className="absolute top-0 right-0 -mt-4 -mr-4 w-24 h-24 bg-white/10 rounded-full blur-xl animate-pulse"></div>
                    <div className="flex justify-between items-start">
                        <p className="text-blue-100 mb-1 font-medium">{currentMarket.name} Balance</p>
                        <span className="px-2 py-1 bg-white/20 rounded text-xs font-bold backdrop-blur-sm">{currentMarket.symbol}</span>
                    </div>
                    <h3 className="text-4xl font-bold tracking-tight">{currentMarket.currency}{balance.toFixed(2)}</h3>
                </div>

                {/* Actions */}
                <div className="mb-6 space-y-3">
                    <div className="flex space-x-3">
                        <div className="relative flex-1">
                            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 font-medium">{currentMarket.currency}</span>
                            <input
                                type="number"
                                value={amount}
                                onChange={(e) => {
                                    setAmount(e.target.value);
                                    if (error) setError(null);
                                }}
                                placeholder="Amount"
                                className={`w-full pl-8 pr-4 py-2 border rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all ${error ? 'border-red-400 bg-red-50 focus:ring-red-500' : 'border-gray-300'
                                    }`}
                                min="0"
                                step="0.01"
                            />
                        </div>

                        <button
                            onClick={() => handleTransactionClick('add')}
                            className="flex items-center justify-center px-4 py-2 bg-green-500 hover:bg-green-600 text-white rounded-xl font-medium transition-colors"
                        >
                            <Plus className="w-4 h-4 mr-1" /> Add
                        </button>

                        <button
                            onClick={() => handleTransactionClick('withdraw')}
                            className="flex items-center justify-center px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white rounded-xl font-medium transition-colors"
                        >
                            <Minus className="w-4 h-4 mr-1" /> Withdraw
                        </button>
                    </div>

                    {error && (
                        <div className="animate-in slide-in-from-top-1 fade-in text-sm text-red-600 flex items-center bg-red-50 p-2 rounded-lg border border-red-100">
                            <X className="w-4 h-4 mr-2 text-red-500 flex-shrink-0" />
                            {error}
                        </div>
                    )}
                </div>

                {/* Transactions Section */}
                <div className="flex items-center justify-between mb-4 mt-2">
                    <h4 className="font-semibold text-gray-800 flex items-center">
                        <History className="w-4 h-4 mr-2 text-gray-500" />
                        Recent Transactions
                    </h4>
                    <button
                        onClick={downloadPDF}
                        disabled={marketTransactions.length === 0}
                        className={`flex items-center text-sm font-medium px-3 py-1.5 rounded-lg transition-colors ${marketTransactions.length > 0
                            ? 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            : 'bg-gray-50 text-gray-400 cursor-not-allowed'
                            }`}
                        title="Download as PDF"
                    >
                        <Download className="w-4 h-4 mr-1" /> PDF
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto border border-gray-100 rounded-xl bg-gray-50 p-2 space-y-2 min-h-[150px]">
                    {marketTransactions.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-2 py-8">
                            <History className="w-8 h-8 opacity-50" />
                            <p className="text-sm">No transactions yet</p>
                        </div>
                    ) : (
                        marketTransactions.map(t => (
                            <div key={t.id} className="bg-white p-3 rounded-lg border border-gray-100 shadow-sm flex items-center justify-between">
                                <div className="flex items-center space-x-3">
                                    <div className={`p-2 rounded-full ${t.type === 'add' ? 'bg-green-100 text-green-600' : 'bg-orange-100 text-orange-600'}`}>
                                        {t.type === 'add' ? <Plus className="w-4 h-4" /> : <Minus className="w-4 h-4" />}
                                    </div>
                                    <div>
                                        <p className="text-sm font-medium text-gray-800">
                                            {t.type === 'add' ? 'Deposit' : 'Withdrawal'}
                                        </p>
                                        <p className="text-xs text-gray-400">{t.date}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className={`text-sm font-bold ${t.type === 'add' ? 'text-green-600' : 'text-gray-800'}`}>
                                        {t.type === 'add' ? '+' : '-'}{currentMarket.currency}{t.amount.toFixed(2)}
                                    </p>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
}
