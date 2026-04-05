"use client";
import { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User } from 'lucide-react';
import axios from 'axios';
import { formatCurrency } from '../utils/currency';

const POPULAR_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "L&T", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "ONGC", "NTPC",
    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "INTC", "AMD", "BTC", "ETH",
    "NIFTY", "BANKNIFTY", "SENSEX", "GOLD", "SILVER"
];

export default function AIAssistant() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { id: 1, text: "Hi! I'm your AI Market Assistant. I'm ready for any question! 🧠\nTry 'Financials of TCS', 'Analyze Reliance', or 'What is the PE of Apple?'", sender: 'bot' }
    ]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isOpen]);

    const processMessage = async (text) => {
        const lowerText = text.toLowerCase();
        let ticker = null;
        let contextData = "";

        // 1. Check for Ticker Mention for Context
        const cleanText = lowerText.replace(/[^a-z0-9\s.]/g, '');
        const words = cleanText.split(/\s+/);

        for (const word of words) {
            const upperWord = word.toUpperCase();
            if (POPULAR_TICKERS.includes(upperWord)) {
                ticker = upperWord;
                break;
            }
        }

        // If a ticker is found, try to fetch current real-time data to pass as context
        if (ticker) {
            try {
                let searchTicker = ticker;
                if (!ticker.includes('.') && !['AAPL', 'TSLA', 'GOOG', 'MSFT', 'AMZN', 'BTC', 'ETH', 'USD', 'EUR', 'NVDA', 'META', 'NFLX', 'NIFTY', 'BANKNIFTY', 'SENSEX'].includes(ticker)) {
                    searchTicker += '.NS';
                }

                if (ticker === 'NIFTY') searchTicker = '^NSEI';
                if (ticker === 'SENSEX') searchTicker = '^BSESN';
                if (ticker === 'BANKNIFTY') searchTicker = '^NSEBANK';

                const response = await axios.get(`/api/v1/stock/${searchTicker}`, {
                    params: { period: '1mo', interval: '1d' }
                });

                const info = response.data.info;
                contextData = `Real-time data for ${ticker}: Price: ${info.currency} ${info.currentPrice || info.regularMarketPrice}, P/E: ${info.peRatio}, Market Cap: ${info.marketCap}, 52W High: ${info.fiftyTwoWeekHigh}, Recommendation: ${info.recommendationKey}. Sector: ${info.sector}.`;
            } catch (error) {
                console.log("Could not fetch real-time context, relying strictly on AI knowledge");
            }
        }

        // Call the new AI Endpoint
        try {
            const response = await axios.post('/api/v1/chat/', {
                message: text,
                context: contextData
            });
            return response.data.response;
        } catch (error) {
            console.error("AI Chat API Error", error);
            // Fallback gracefully
            return "I apologize, but I am currently having trouble connecting to my AI core. Please try again in a moment.";
        }
    };

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        const userMsg = { id: Date.now(), text: input, sender: 'user' };
        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setIsTyping(true);

        // Process message
        const answer = await processMessage(userMsg.text);

        const botMsg = { id: Date.now() + 1, text: answer, sender: 'bot' };
        setMessages(prev => [...prev, botMsg]);
        setIsTyping(false);
    };

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end font-sans">
            {/* Chat Window */}
            {isOpen && (
                <div className="mb-4 w-80 md:w-96 bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden animate-in slide-in-from-bottom-10 fade-in duration-300 flex flex-col h-[500px]">
                    {/* Header */}
                    <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-4 flex justify-between items-center text-white">
                        <div className="flex items-center space-x-2">
                            <div className="bg-white/20 p-1.5 rounded-lg">
                                <Bot className="w-5 h-5" />
                            </div>
                            <div>
                                <h3 className="font-bold text-sm">Market Assistant</h3>
                                <div className="flex items-center space-x-1">
                                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                                    <span className="text-xs text-blue-100">Online</span>
                                </div>
                            </div>
                        </div>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="text-white/80 hover:text-white hover:bg-white/10 p-1 rounded-lg transition-colors"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Messages Area */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
                        {messages.map((msg) => (
                            <div
                                key={msg.id}
                                className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                            >
                                {msg.sender === 'bot' && (
                                    <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2 flex-shrink-0 border border-blue-200">
                                        <Bot className="w-4 h-4 text-blue-600" />
                                    </div>
                                )}
                                <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-sm ${msg.sender === 'user'
                                    ? 'bg-blue-600 text-white rounded-tr-none'
                                    : 'bg-white text-gray-800 border border-gray-100 rounded-tl-none prose prose-sm max-w-none'
                                    }`}
                                    dangerouslySetInnerHTML={msg.sender === 'bot' ? { __html: msg.text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br/>') } : undefined}
                                >
                                    {msg.sender === 'user' ? msg.text : undefined}
                                </div>
                            </div>
                        ))}
                        {isTyping && (
                            <div className="flex justify-start">
                                <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-none px-4 py-3 shadow-sm">
                                    <div className="flex space-x-1">
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-white border-t border-gray-100">
                        <form onSubmit={handleSend} className="flex items-center space-x-2">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask about stocks..."
                                className="flex-1 bg-gray-50 border border-gray-200 text-gray-900 text-sm rounded-xl focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 outline-none transition-all"
                            />
                            <button
                                type="submit"
                                disabled={!input.trim() || isTyping}
                                className="bg-blue-600 text-white p-2.5 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-md shadow-blue-600/20"
                            >
                                <Send className="w-4 h-4" />
                            </button>
                        </form>
                    </div>
                </div>
            )}

            {/* Toggle Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`p-4 rounded-full shadow-lg transition-all duration-300 transform hover:scale-105 ${isOpen
                    ? 'bg-gray-800 text-white rotate-90'
                    : 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-blue-600/30'
                    }`}
            >
                {isOpen ? <X className="w-6 h-6" /> : <MessageCircle className="w-6 h-6" />}
            </button>
        </div>
    );
}
