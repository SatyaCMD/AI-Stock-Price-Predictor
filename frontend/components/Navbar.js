"use client";
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';
import { TrendingUp, BarChart2, PieChart, Settings, Menu, X, Wallet, BookOpen, LogOut, User, Shield } from 'lucide-react';

import WalletModal from './WalletModal';
import DemoWallet from './DemoWallet';
import PremiumModal from './PremiumModal';
import { toast } from 'react-hot-toast';

export default function Navbar() {
    const pathname = usePathname();
    const [isOpen, setIsOpen] = useState(false);
    const [showSettings, setShowSettings] = useState(false);
    const [walletAddress, setWalletAddress] = useState(null);
    const [showWalletModal, setShowWalletModal] = useState(false);
    const [showDemoWallet, setShowDemoWallet] = useState(false);
    const [showPremiumModal, setShowPremiumModal] = useState(false);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [userProfile, setUserProfile] = useState({ name: 'User', email: 'user@example.com' });
    const [pendingApprovals, setPendingApprovals] = useState(0);
    const connectWallet = () => {
        setShowWalletModal(true);
    };

    const handleMetaMaskConnect = async () => {
        if (typeof window.ethereum !== 'undefined') {
            try {
                const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
                setWalletAddress(accounts[0]);
                localStorage.setItem('walletAddress', accounts[0]);
                setShowWalletModal(false);
            } catch (error) {
                if (error.code === 4001) {
                    console.log("User rejected wallet connection");
                    toast.error("Connection rejected. Please try again.");
                } else {
                    console.error("Wallet connection error", error);
                    toast.error("Failed to connect wallet.");
                }
            }
        } else {
            toast.error("MetaMask is not installed. Please install it to use this feature.");
        }
    };

    const handleMockConnect = () => {
        const mockAddress = "0xDemoWallet...0000";
        setWalletAddress(mockAddress);
        localStorage.setItem('walletAddress', mockAddress);
        setShowWalletModal(false);
        setShowDemoWallet(true);
    };

    const disconnectWallet = () => {
        setWalletAddress(null);
        localStorage.removeItem('walletAddress');
    };

    // Check for connected wallet on mount
    useEffect(() => {
        if (typeof window !== 'undefined') {
            const savedAddress = localStorage.getItem('walletAddress');
            if (savedAddress) {
                setWalletAddress(savedAddress);
            }
        }
    }, []);

    // Listen for login/logout events
    useEffect(() => {
        const checkAuth = () => {
            const loggedIn = localStorage.getItem('isLoggedIn') === 'true';
            setIsLoggedIn(loggedIn);

            if (loggedIn) {
                const profile = JSON.parse(localStorage.getItem('userProfile') || '{}');
                
                // --- Auto Approval Logic (5 mins) ---
                if (profile.paymentPending && profile.paymentTimestamp) {
                    const FIVE_MINUTES = 5 * 60 * 1000;
                    if (Date.now() - profile.paymentTimestamp > FIVE_MINUTES) {
                        profile.paymentPending = false;
                        profile.isSubscribed = true;
                        
                        let durationDays = 30;
                        if (profile.paymentPendingType === 'yearly') durationDays = 365;
                        if (profile.paymentPendingType === 'quarterly') durationDays = 90;
            
                        profile.subscriptionStartDate = new Date().toISOString();
                        profile.subscriptionNextDate = new Date(Date.now() + durationDays * 24 * 60 * 60 * 1000).toISOString();
                        profile.subscriptionPlan = profile.paymentPendingType || 'monthly';
                        
                        localStorage.setItem('userProfile', JSON.stringify(profile));
            
                        const registeredUsers = JSON.parse(localStorage.getItem('registeredUsers') || '[]');
                        const userIndex = registeredUsers.findIndex(u => u.email === profile.email);
                        if (userIndex !== -1) {
                            registeredUsers[userIndex] = { ...registeredUsers[userIndex], ...profile };
                            localStorage.setItem('registeredUsers', JSON.stringify(registeredUsers));
                        }
                    }
                }
                // ------------------------------------

                // --- Auto Approval Logic (KYC) ---
                if (profile.kycStatus === 'pending' && profile.kycSubmittedAt) {
                    const TEN_MINUTES = 10 * 60 * 1000;
                    if (Date.now() - profile.kycSubmittedAt > TEN_MINUTES) {
                        profile.kycStatus = 'verified';
                        profile.kycVerified = true;
                        profile.kycDetails = { ...profile.kycDetails, verifiedDate: new Date().toISOString() };
                        localStorage.setItem('userProfile', JSON.stringify(profile));
                        
                        const registeredUsers = JSON.parse(localStorage.getItem('registeredUsers') || '[]');
                        const userIndex = registeredUsers.findIndex(u => u.email === profile.email);
                        if (userIndex !== -1) {
                            registeredUsers[userIndex].kycStatus = 'verified';
                            registeredUsers[userIndex].kycVerified = true;
                            registeredUsers[userIndex].kycDetails = profile.kycDetails;
                            localStorage.setItem('registeredUsers', JSON.stringify(registeredUsers));
                        }
                    }
                }
                // ---------------------------------

                setUserProfile(profile);
                
                // Load global users to find payment/KYC pending tasks for admin badge
                if (profile.role === 'admin') {
                    const users = JSON.parse(localStorage.getItem('registeredUsers') || '[]');
                    const pending = users.filter(u => u.paymentPending || u.kycStatus === 'pending');
                    setPendingApprovals(pending.length);
                }
            } else {
                setUserProfile(null);
                setPendingApprovals(0);
            }
        };

        checkAuth();
        const intervalId = setInterval(checkAuth, 30000); // Check every 30s
        window.addEventListener('authChange', checkAuth);
        return () => {
            window.removeEventListener('authChange', checkAuth);
            clearInterval(intervalId);
        };
    }, []);

    return (
        <>
            <WalletModal
                isOpen={showWalletModal}
                onClose={() => setShowWalletModal(false)}
                onConnectMock={handleMockConnect}
                onConnectMetaMask={handleMetaMaskConnect}
            />
            <DemoWallet
                isOpen={showDemoWallet}
                onClose={() => setShowDemoWallet(false)}
            />
            <PremiumModal
                isOpen={showPremiumModal}
                onClose={() => setShowPremiumModal(false)}
                userProfile={userProfile}
            />
            <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-xl border-b border-gray-200 supports-[backdrop-filter]:bg-white/60">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        {/* Logo */}
                        <div className="flex items-center flex-shrink-0">
                            <Link href="/" className="flex items-center space-x-2 group">
                                <div className="bg-gradient-to-tr from-blue-600 to-indigo-600 p-2 rounded-lg group-hover:scale-105 transition-transform shadow-md shadow-blue-500/20">
                                    <TrendingUp className="h-6 w-6 text-white" />
                                </div>
                                <span className="text-xl font-bold text-gray-900 tracking-tight">
                                    TradeMind AI
                                </span>
                            </Link>
                        </div>

                        {/* Desktop Menu */}
                        <div className="hidden md:block">
                            <div className="ml-10 flex items-baseline space-x-8">
                                <Link
                                    href="/"
                                    className={`px-3 py-2 rounded-md text-sm font-medium transition-all ${pathname === '/'
                                        ? 'text-gray-900 border-b-2 border-blue-600'
                                        : 'text-gray-500 hover:text-blue-600 hover:bg-blue-50'
                                        }`}
                                >
                                    Dashboard
                                </Link>
                                <Link
                                    href="/markets"
                                    className={`px-3 py-2 rounded-md text-sm font-medium transition-all ${pathname === '/markets'
                                        ? 'text-gray-900 border-b-2 border-blue-600'
                                        : 'text-gray-500 hover:text-blue-600 hover:bg-blue-50'
                                        }`}
                                >
                                    Markets
                                </Link>
                                <Link
                                    href="/portfolio"
                                    className={`px-3 py-2 rounded-md text-sm font-medium transition-all ${pathname === '/portfolio'
                                        ? 'text-gray-900 border-b-2 border-blue-600'
                                        : 'text-gray-500 hover:text-blue-600 hover:bg-blue-50'
                                        }`}
                                >
                                    Portfolio
                                </Link>
                                <Link
                                    href="/learn"
                                    className={`px-3 py-2 rounded-md text-sm font-medium transition-all ${pathname === '/learn'
                                        ? 'text-gray-900 border-b-2 border-blue-600'
                                        : 'text-gray-500 hover:text-blue-600 hover:bg-blue-50'
                                        }`}
                                >
                                    Learn
                                </Link>
                            </div>
                        </div>

                        {/* Right Side Actions */}
                        <div className="hidden md:flex items-center space-x-4">
                            {/* Upgrade Button */}
                            {isLoggedIn && !userProfile.isSubscribed && (
                                <Link
                                    href="/upgrade"
                                    className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-bold transition-all shadow-md ${userProfile.paymentPending ? 'bg-yellow-100 text-yellow-800 border border-yellow-300' : 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:shadow-lg hover:scale-105'}`}
                                >
                                    <span>{userProfile.paymentPending ? 'Approval Pending' : 'Upgrade Now'}</span>
                                </Link>
                            )}
                            
                            {isLoggedIn && userProfile.isSubscribed && (
                                <button
                                    onClick={() => setShowPremiumModal(true)}
                                    className="flex items-center space-x-2 px-3 py-1.5 bg-gradient-to-r from-yellow-400 to-amber-500 text-yellow-950 rounded-full text-xs font-bold shadow-sm border border-yellow-300 hover:shadow-md hover:scale-105 transition-all"
                                >
                                    <Shield className="w-3 h-3" />
                                    <span>Premium</span>
                                </button>
                            )}

                            {isLoggedIn ? (
                                <div className="relative">
                                    <button
                                        onClick={() => setShowSettings(!showSettings)}
                                        className="flex items-center space-x-2 p-1 pr-3 rounded-full border border-gray-200 hover:bg-gray-50 transition-all"
                                    >
                                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold shadow-sm">
                                            <User className="w-4 h-4" />
                                        </div>
                                        <span className="text-sm font-medium text-gray-700">Profile</span>
                                    </button>
                                    {/* Profile Dropdown */}
                                    {showSettings && (
                                        <div className="absolute right-0 mt-2 w-56 bg-white rounded-xl shadow-xl border border-gray-100 py-2 animate-in origin-top-right z-50">
                                            <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/50">
                                                <p className="text-sm font-bold text-gray-900">{userProfile.name}</p>
                                                <p className="text-xs text-gray-500 truncate">{userProfile.email}</p>
                                            </div>
                                            <div className="py-1">
                                                <Link href="/profile" className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors flex items-center" onClick={() => setShowSettings(false)}>
                                                    <User className="w-4 h-4 mr-2 text-gray-400" />
                                                    Profile
                                                </Link>
                                                {userProfile.role === 'admin' && (
                                                    <Link href="/admin" className="w-full text-left px-4 py-2 text-sm text-purple-700 hover:bg-purple-50 transition-colors flex items-center" onClick={() => setShowSettings(false)}>
                                                        <Shield className="w-4 h-4 mr-2 text-purple-500" />
                                                        Admin Dashboard
                                                    </Link>
                                                )}
                                            </div>
                                            <div className="border-t border-gray-100 py-1">
                                                <button
                                                    onClick={() => {
                                                        setShowSettings(false);
                                                        localStorage.removeItem('isLoggedIn');
                                                        setIsLoggedIn(false);
                                                        window.location.href = '/login';
                                                    }}
                                                    className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors flex items-center"
                                                >
                                                    <LogOut className="w-4 h-4 mr-2" />
                                                    Logout
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <Link
                                    href="/login"
                                    className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all shadow-lg shadow-blue-600/20"
                                >
                                    <User className="h-4 w-4" />
                                    <span>Login</span>
                                </Link>
                            )}

                            {
                                walletAddress ? (
                                    <div className="flex items-center space-x-2">
                                        <button
                                            onClick={() => walletAddress.startsWith("0xDemoWallet") ? setShowDemoWallet(true) : null}
                                            className={`flex items-center space-x-2 ${walletAddress.startsWith("0xDemoWallet") ? "hover:bg-green-100 cursor-pointer transition-colors" : ""} bg-green-50 text-green-700 border border-green-200 px-4 py-2 rounded-lg text-sm font-medium`}
                                        >
                                            <Wallet className="h-4 w-4" />
                                            <span>{walletAddress.startsWith("0xDemoWallet") ? "Demo Wallet" : `${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`}</span>
                                        </button>
                                        <button
                                            onClick={disconnectWallet}
                                            title="Unlink Wallet"
                                            className="p-2 bg-red-50 text-red-600 border border-red-200 rounded-lg hover:bg-red-100 transition-colors"
                                        >
                                            <LogOut className="h-4 w-4" />
                                        </button>
                                    </div>
                                ) : (
                                    <button
                                        onClick={connectWallet}
                                        className="flex items-center space-x-2 bg-gray-900 hover:bg-gray-800 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all shadow-lg shadow-gray-900/10 hover:shadow-gray-900/20"
                                    >
                                        <Wallet className="h-4 w-4" />
                                        <span>Connect Wallet</span>
                                    </button>
                                )
                            }
                        </div >

                        {/* Mobile menu button */}
                        < div className="md:hidden flex items-center" >
                            <button
                                onClick={() => setIsOpen(!isOpen)}
                                className="inline-flex items-center justify-center p-2 rounded-md text-gray-500 hover:text-gray-900 hover:bg-gray-100 focus:outline-none"
                            >
                                {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                            </button>
                        </div >
                    </div >
                </div >

                {/* Mobile Menu */}
                {
                    isOpen && (
                        <div className="md:hidden bg-white border-b border-gray-200 animate-in slide-in-from-top-5">
                            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
                                <Link
                                    href="/"
                                    className={`block px-3 py-2 rounded-md text-base font-medium ${pathname === '/'
                                        ? 'text-blue-600 bg-blue-50'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                        }`}
                                >
                                    Dashboard
                                </Link>
                                <Link
                                    href="/markets"
                                    className={`block px-3 py-2 rounded-md text-base font-medium ${pathname === '/markets'
                                        ? 'text-blue-600 bg-blue-50'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                        }`}
                                >
                                    Markets
                                </Link>
                                <Link
                                    href="/portfolio"
                                    className={`block px-3 py-2 rounded-md text-base font-medium ${pathname === '/portfolio'
                                        ? 'text-blue-600 bg-blue-50'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                        }`}
                                >
                                    Portfolio
                                </Link>
                                <Link
                                    href="/learn"
                                    className={`block px-3 py-2 rounded-md text-base font-medium ${pathname === '/learn'
                                        ? 'text-blue-600 bg-blue-50'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                        }`}
                                >
                                    Learn
                                </Link>
                                {walletAddress ? (
                                    <div className="mt-4 space-y-2">
                                        <button
                                            onClick={() => walletAddress.startsWith("0xDemoWallet") ? setShowDemoWallet(true) : null}
                                            className={`w-full flex items-center justify-center space-x-2 ${walletAddress.startsWith("0xDemoWallet") ? "hover:bg-green-100 transition-colors" : ""} bg-green-50 text-green-700 border border-green-200 px-4 py-2 rounded-lg text-base font-medium`}
                                        >
                                            <Wallet className="h-4 w-4" />
                                            <span>{walletAddress.startsWith("0xDemoWallet") ? "Demo Wallet" : `${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`}</span>
                                        </button>
                                        <button
                                            onClick={disconnectWallet}
                                            className="w-full flex items-center justify-center space-x-2 bg-red-50 text-red-600 border border-red-200 px-4 py-2 rounded-lg text-base font-medium hover:bg-red-100"
                                        >
                                            <LogOut className="h-4 w-4" />
                                            <span>Unlink Wallet</span>
                                        </button>
                                    </div>
                                ) : (
                                    <button
                                        onClick={connectWallet}
                                        className="w-full mt-4 flex items-center justify-center space-x-2 bg-gray-900 text-white px-4 py-2 rounded-lg text-base font-medium"
                                    >
                                        <Wallet className="h-4 w-4" />
                                        <span>Connect Wallet</span>
                                    </button>
                                )}
                            </div>
                        </div>
                    )
                }
            </nav >
        </>
    );
}
