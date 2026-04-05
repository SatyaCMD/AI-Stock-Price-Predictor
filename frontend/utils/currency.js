export const formatCurrency = (value, currencyCode) => {
    if (value === undefined || value === null) return 'N/A';

    // Map common currency codes to locales
    const localeMap = {
        'INR': 'en-IN',
        'USD': 'en-US',
        'JPY': 'ja-JP',
        'EUR': 'de-DE',
        'GBP': 'en-GB',
        'CNY': 'zh-CN',
    };

    // Default to US if not found
    const locale = localeMap[currencyCode] || 'en-US';

    // Special handling for Crypto/Forex which might come as 'USD' but we want to ensure it's treated as such
    // The user requested Crypto/Forex to be in USD.
    // If the API returns 'USD' it will be handled by the default formatter.

    try {
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency: currencyCode || 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    } catch (error) {
        console.error("Currency formatting error:", error);
        return `${currencyCode} ${value.toFixed(2)}`;
    }
};
