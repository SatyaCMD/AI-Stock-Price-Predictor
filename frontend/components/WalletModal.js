"use client";
import { X, Wallet, ExternalLink, Shield } from 'lucide-react';

export default function WalletModal({ isOpen, onClose, onConnectMock, onConnectMetaMask }) {
    if (!isOpen) return null;

    const hasMetaMask = typeof window !== 'undefined' && typeof window.ethereum !== 'undefined';

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-in fade-in duration-200">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/20 backdrop-blur-sm transition-opacity"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative w-full max-w-md bg-white/90 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/50 p-6 transform transition-all scale-100">
                {/* Close Button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full text-gray-400 hover:text-gray-600 hover:bg-gray-100/50 transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>

                {/* Header */}
                <div className="text-center mb-6">
                    <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4 shadow-inner">
                        <Wallet className="w-8 h-8 text-blue-600" />
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">Connect Wallet</h2>
                    <p className="text-gray-600">
                        Choose how you want to connect to TradeMind AI.
                    </p>
                </div>

                {/* Actions */}
                <div className="space-y-3">
                    {hasMetaMask ? (
                        <button
                            onClick={onConnectMetaMask}
                            className="flex items-center justify-between w-full p-4 bg-orange-50 hover:bg-orange-100 border border-orange-200 rounded-xl transition-all group"
                        >
                            <div className="flex items-center space-x-3">
                                <div className="p-2 bg-white rounded-lg shadow-sm">
                                    <img src="https://upload.wikimedia.org/wikipedia/commons/3/36/MetaMask_Fox.svg" alt="MetaMask" className="w-6 h-6" />
                                </div>
                                <div className="text-left">
                                    <h3 className="font-semibold text-gray-900 group-hover:text-orange-700">MetaMask</h3>
                                    <p className="text-xs text-gray-500">Connect to your Web3 wallet</p>
                                </div>
                            </div>
                            <Wallet className="w-5 h-5 text-gray-400 group-hover:text-orange-600" />
                        </button>
                    ) : (
                        <a
                            href="https://metamask.io/download/"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center justify-between w-full p-4 bg-orange-50 hover:bg-orange-100 border border-orange-200 rounded-xl transition-all group"
                        >
                            <div className="flex items-center space-x-3">
                                <div className="p-2 bg-white rounded-lg shadow-sm">
                                    <img src="https://upload.wikimedia.org/wikipedia/commons/3/36/MetaMask_Fox.svg" alt="MetaMask" className="w-6 h-6" />
                                </div>
                                <div className="text-left">
                                    <h3 className="font-semibold text-gray-900 group-hover:text-orange-700">Install MetaMask</h3>
                                    <p className="text-xs text-gray-500">Connect your real wallet</p>
                                </div>
                            </div>
                            <ExternalLink className="w-5 h-5 text-gray-400 group-hover:text-orange-600" />
                        </a>
                    )}

                    <button
                        onClick={onConnectMock}
                        className="flex items-center justify-between w-full p-4 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-xl transition-all group"
                    >
                        <div className="flex items-center space-x-3">
                            <div className="p-2 bg-white rounded-lg shadow-sm">
                                <Shield className="w-6 h-6 text-blue-600" />
                            </div>
                            <div className="text-left">
                                <h3 className="font-semibold text-gray-900 group-hover:text-blue-700">Demo Mode</h3>
                                <p className="text-xs text-gray-500">Continue with mock wallet</p>
                            </div>
                        </div>
                        <Wallet className="w-5 h-5 text-gray-400 group-hover:text-blue-600" />
                    </button>
                </div>

                {/* Footer */}
                <div className="mt-6 text-center">
                    <p className="text-xs text-gray-400">
                        By connecting, you agree to our Terms of Service and Privacy Policy.
                    </p>
                </div>
            </div>
        </div>
    );
}
