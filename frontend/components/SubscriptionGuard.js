"use client";
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Lock, Clock, CreditCard, ChevronRight } from 'lucide-react';

export default function SubscriptionGuard({ children }) {
    const router = useRouter();
    const [isLoading, setIsLoading] = useState(true);
    const [hasAccess, setHasAccess] = useState(false);
    const [trialTimeLeft, setTrialTimeLeft] = useState('');
    const [isTrial, setIsTrial] = useState(false);
    const [isPending, setIsPending] = useState(false);

    useEffect(() => {
        const checkAccess = () => {
            const profileStr = localStorage.getItem('userProfile');
            if (!profileStr) {
                // Not logged in
                router.push('/login');
                return;
            }

            const profile = JSON.parse(profileStr);
            
            // Admins always have access
            if (profile.role === 'admin') {
                setHasAccess(true);
                setIsLoading(false);
                return;
            }

            // If subscribed, they have access
            if (profile.isSubscribed) {
                setHasAccess(true);
                setIsTrial(false);
                setIsLoading(false);
                return;
            }

            // Check trial
            if (profile.trialStart) {
                const start = new Date(profile.trialStart);
                const now = new Date();
                const diffMs = now - start;
                const hoursLeft = 24 - (diffMs / (1000 * 60 * 60));

                if (hoursLeft > 0) {
                    setHasAccess(true);
                    setIsTrial(true);
                    
                    if (hoursLeft > 1) {
                        setTrialTimeLeft(`${Math.floor(hoursLeft)} hours`);
                    } else {
                        const minsLeft = Math.floor(hoursLeft * 60);
                        setTrialTimeLeft(`${minsLeft} minutes`);
                    }
                    setIsLoading(false);
                    return;
                }
            }

            // Check if pending
            if (profile.paymentPending) {
                setIsPending(true);
            } else {
                setIsPending(false);
            }

            // Trial expired and not subscribed
            setHasAccess(false);
            setIsLoading(false);
        };

        checkAccess();
        window.addEventListener('authChange', checkAccess);
        return () => window.removeEventListener('authChange', checkAccess);
    }, [router]);

    if (isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    if (!hasAccess) {
        return (
            <div className="min-h-screen pt-24 pb-12 px-4 sm:px-6 flex items-center justify-center bg-gray-50/50">
                <div className="max-w-md w-full glass-card p-8 rounded-2xl text-center relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/10 rounded-full blur-3xl -mr-16 -mt-16"></div>
                    <div className="absolute bottom-0 left-0 w-32 h-32 bg-blue-500/10 rounded-full blur-3xl -ml-16 -mb-16"></div>
                    
                    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg ${isPending ? 'bg-gradient-to-br from-yellow-400 to-yellow-600 shadow-yellow-500/20' : 'bg-gradient-to-br from-red-500 to-rose-600 shadow-red-500/20'}`}>
                        {isPending ? <Clock className="w-8 h-8 text-white animate-pulse" /> : <Lock className="w-8 h-8 text-white" />}
                    </div>
                    
                    <h2 className="text-2xl font-bold text-gray-900 mb-2">{isPending ? 'Approval Pending' : 'Subscription Required'}</h2>
                    <p className="text-gray-600 mb-6 leading-relaxed">
                        {isPending 
                            ? "Your payment is currently pending review. Please wait for an Administrator to verify your transaction and grant you Premium Access."
                            : "Your 1-day free trial has expired. To continue accessing premium AI predictions, portfolio features, and in-depth analysis, please activate your subscription."}
                    </p>

                    <div className={`p-4 rounded-xl text-sm font-medium border mb-6 flex items-start gap-3 text-left ${isPending ? 'bg-yellow-50 text-yellow-800 border-yellow-200' : 'bg-red-50 text-red-700 border-red-100'}`}>
                        <CreditCard className="w-5 h-5 flex-shrink-0 mt-0.5" />
                        <div>
                            <p className="font-bold mb-1">{isPending ? 'Waiting for Admin' : 'Payment Action Required'}</p>
                            <p className={isPending ? "text-yellow-700 font-normal" : "text-red-600/90 font-normal"}>
                                {isPending ? "No further action is required from you. You will be granted access automatically once approved." : "Please complete your subscription payment to restore access to all features."}
                            </p>
                        </div>
                    </div>

                    <div className="space-y-3">
                        {!isPending && (
                            <Link href="/upgrade" className="w-full bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white font-semibold py-3 px-4 rounded-xl transition-all shadow-lg hover:shadow-xl flex items-center justify-center gap-2 mb-3">
                                Upgrade Now
                                <ChevronRight className="w-4 h-4" />
                            </Link>
                        )}
                        <button 
                            onClick={() => router.push('/')}
                            className="w-full bg-white hover:bg-gray-50 text-gray-700 font-medium py-3 px-4 rounded-xl border border-gray-200 transition-colors"
                        >
                            Return to Dashboard
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <>
            {isTrial && (
                <div className="bg-gradient-to-r from-amber-500 to-orange-600 text-white px-4 py-2 text-center text-sm font-medium fixed bottom-0 w-full z-50 flex items-center justify-center gap-2 shadow-lg animate-in slide-in-from-bottom-5">
                    <Clock className="w-4 h-4" />
                    <span>Free Trial Active — {trialTimeLeft} remaining</span>
                    <button onClick={() => router.push('/upgrade')} className="ml-2 underline hover:text-amber-100 transition-colors font-bold">Upgrade Now</button>
                </div>
            )}
            {children}
        </>
    );
}
