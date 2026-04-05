"use client";
import React, { useState, useEffect } from 'react';
import { X, Save, Lock, Bell, User, Check, Phone } from 'lucide-react';

export default function SettingsModal({ isOpen, onClose, user, onUpdateProfile }) {
    const [activeTab, setActiveTab] = useState('profile');
    const [formData, setFormData] = useState({
        name: user.name || '',
        email: user.email || '',
        mobile: user?.kycDetails?.mobile || '',
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
        emailNotifications: true,
        pushNotifications: false
    });
    const [message, setMessage] = useState({ type: '', text: '' });
    
    // OTP logic for contact updates
    const [contactOtpMode, setContactOtpMode] = useState(false);
    const [debugOtp, setDebugOtp] = useState('');
    const [userOtpInput, setUserOtpInput] = useState('');

    useEffect(() => {
        if (isOpen) {
            setFormData(prev => ({ 
                ...prev, 
                name: user.name || '',
                email: user.email || '',
                mobile: user?.kycDetails?.mobile || ''
            }));
            setMessage({ type: '', text: '' });
            setContactOtpMode(false);
        }
    }, [isOpen, user]);

    if (!isOpen) return null;

    const handleChange = (e) => {
        const value = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
        setFormData({ ...formData, [e.target.name]: value });
    };

    const handleSave = (e) => {
        e.preventDefault();
        setMessage({ type: '', text: '' });

        if (activeTab === 'profile') {
            if (!formData.name.trim()) {
                setMessage({ type: 'error', text: 'Name cannot be empty' });
                return;
            }
            onUpdateProfile({ name: formData.name });
            setMessage({ type: 'success', text: 'Profile updated successfully!' });
        } else if (activeTab === 'contact') {
            if (!contactOtpMode) {
                // Generate OTP to Verify
                const newOtp = Math.floor(100000 + Math.random() * 900000).toString();
                setDebugOtp(newOtp);
                setContactOtpMode(true);
                setMessage({ type: 'success', text: 'Verification OTP sent to your new email/mobile!' });
                return;
            } else {
                // Verify OTP
                if (userOtpInput !== debugOtp) {
                    setMessage({ type: 'error', text: 'Incorrect OTP. Please try again.' });
                    return;
                }
                
                // Update User logic
                const kycUpdated = { ...(user.kycDetails || {}), mobile: formData.mobile };
                onUpdateProfile({ email: formData.email, kycDetails: kycUpdated });
                setMessage({ type: 'success', text: 'Contact details verified & updated!' });
                setContactOtpMode(false);
                setUserOtpInput('');
            }
        } else if (activeTab === 'password') {
            if (!formData.currentPassword || !formData.newPassword || !formData.confirmPassword) {
                setMessage({ type: 'error', text: 'All fields are required' });
                return;
            }
            if (formData.newPassword !== formData.confirmPassword) {
                setMessage({ type: 'error', text: 'New passwords do not match' });
                return;
            }
            // Mock password validation
            if (formData.newPassword.length < 6) {
                setMessage({ type: 'error', text: 'Password must be at least 6 characters' });
                return;
            }
            // In a real app, we would verify currentPassword with backend
            setMessage({ type: 'success', text: 'Password changed successfully!' });
            setFormData(prev => ({ ...prev, currentPassword: '', newPassword: '', confirmPassword: '' }));
        } else if (activeTab === 'notifications') {
            // Save preferences to localStorage
            const prefs = {
                email: formData.emailNotifications,
                push: formData.pushNotifications
            };
            localStorage.setItem('notificationPrefs', JSON.stringify(prefs));
            setMessage({ type: 'success', text: 'Preferences saved!' });
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-in fade-in">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md overflow-hidden">
                <div className="flex items-center justify-between p-6 border-b border-gray-100">
                    <h2 className="text-xl font-bold text-gray-900">Settings</h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 transition-colors">
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <div className="flex border-b border-gray-100">
                    <button
                        onClick={() => { setActiveTab('profile'); setContactOtpMode(false); }}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'profile' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                    >
                        <User className="w-4 h-4 inline-block mr-2" />
                        Profile
                    </button>
                    <button
                        onClick={() => { setActiveTab('contact'); setContactOtpMode(false); }}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'contact' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                    >
                        <Phone className="w-4 h-4 inline-block mr-2" />
                        Contact
                    </button>
                    <button
                        onClick={() => { setActiveTab('password'); setContactOtpMode(false); }}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'password' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                    >
                        <Lock className="w-4 h-4 inline-block mr-2" />
                        Password
                    </button>
                    <button
                        onClick={() => setActiveTab('notifications')}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'notifications' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                    >
                        <Bell className="w-4 h-4 inline-block mr-2" />
                        Notifications
                    </button>
                </div>

                <div className="p-6">
                    {message.text && (
                        <div className={`mb-4 p-3 rounded-lg text-sm flex items-center ${message.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                            {message.type === 'success' ? <Check className="w-4 h-4 mr-2" /> : <X className="w-4 h-4 mr-2" />}
                            {message.text}
                        </div>
                    )}

                    <form onSubmit={handleSave} className="space-y-4">
                        {activeTab === 'profile' && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Display Name</label>
                                <input
                                    type="text"
                                    name="name"
                                    value={formData.name}
                                    onChange={handleChange}
                                    className="w-full px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                                    placeholder="Your Name"
                                />
                            </div>
                        )}

                        {activeTab === 'contact' && (
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
                                    <input
                                        type="email"
                                        name="email"
                                        value={formData.email}
                                        onChange={handleChange}
                                        disabled={contactOtpMode}
                                        className="w-full px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none transition-all disabled:opacity-50"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Mobile Number</label>
                                    <input
                                        type="tel"
                                        name="mobile"
                                        value={formData.mobile}
                                        onChange={handleChange}
                                        disabled={contactOtpMode}
                                        className="w-full px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none transition-all disabled:opacity-50"
                                    />
                                </div>
                                
                                {contactOtpMode && (
                                    <div className="bg-yellow-50 p-4 rounded-xl border border-yellow-200 animate-in fade-in">
                                        <p className="text-sm font-bold text-yellow-800 mb-2">Debug Mode: OTP is <span className="text-xl font-mono ml-2 tracking-widest">{debugOtp}</span></p>
                                        <label className="block text-sm font-medium text-gray-700 mb-1">Enter OTP to Verification</label>
                                        <input
                                            type="text"
                                            maxLength="6"
                                            value={userOtpInput}
                                            onChange={(e) => setUserOtpInput(e.target.value)}
                                            className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-yellow-500 outline-none transition-all font-mono tracking-widest text-center text-xl"
                                            placeholder="••••••"
                                        />
                                    </div>
                                )}
                            </div>
                        )}

                        {activeTab === 'password' && (
                            <>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Current Password</label>
                                    <input
                                        type="password"
                                        name="currentPassword"
                                        value={formData.currentPassword}
                                        onChange={handleChange}
                                        className="w-full px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                                        placeholder="••••••••"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
                                    <input
                                        type="password"
                                        name="newPassword"
                                        value={formData.newPassword}
                                        onChange={handleChange}
                                        className="w-full px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                                        placeholder="••••••••"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Confirm New Password</label>
                                    <input
                                        type="password"
                                        name="confirmPassword"
                                        value={formData.confirmPassword}
                                        onChange={handleChange}
                                        className="w-full px-4 py-2 rounded-lg border border-gray-200 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                                        placeholder="••••••••"
                                    />
                                </div>
                            </>
                        )}

                        {activeTab === 'notifications' && (
                            <div className="space-y-3">
                                <label className="flex items-center space-x-3 p-3 border border-gray-100 rounded-xl hover:bg-gray-50 cursor-pointer transition-colors">
                                    <input
                                        type="checkbox"
                                        name="emailNotifications"
                                        checked={formData.emailNotifications}
                                        onChange={handleChange}
                                        className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500 border-gray-300"
                                    />
                                    <div>
                                        <p className="font-medium text-gray-900">Email Notifications</p>
                                        <p className="text-xs text-gray-500">Receive updates via email</p>
                                    </div>
                                </label>
                                <label className="flex items-center space-x-3 p-3 border border-gray-100 rounded-xl hover:bg-gray-50 cursor-pointer transition-colors">
                                    <input
                                        type="checkbox"
                                        name="pushNotifications"
                                        checked={formData.pushNotifications}
                                        onChange={handleChange}
                                        className="w-5 h-5 text-blue-600 rounded focus:ring-blue-500 border-gray-300"
                                    />
                                    <div>
                                        <p className="font-medium text-gray-900">Push Notifications</p>
                                        <p className="text-xs text-gray-500">Receive updates on your device</p>
                                    </div>
                                </label>
                            </div>
                        )}

                        <button
                            type="submit"
                            className={`w-full text-white font-semibold py-3 rounded-lg transition-colors flex items-center justify-center ${contactOtpMode ? 'bg-yellow-500 hover:bg-yellow-600' : 'bg-blue-600 hover:bg-blue-700'}`}
                        >
                            {contactOtpMode ? <Check className="w-5 h-5 mr-2" /> : <Save className="w-5 h-5 mr-2" />}
                            {contactOtpMode ? 'Verify & Update' : 'Save Changes'}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}
