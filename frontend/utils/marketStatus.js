export const getMarketStatus = (region) => {
    if (region === 'Crypto') return { status: 'Open', detail: '24/7 Market', isOpen: true };

    const now = new Date();
    const utcDay = now.getUTCDay();
    const utcHour = now.getUTCHours();
    const utcMinute = now.getUTCMinutes();
    const utcSecond = now.getUTCSeconds();
    const currentSecondsInWeek = utcDay * 24 * 3600 + utcHour * 3600 + utcMinute * 60 + utcSecond;

    const formatDiff = (secondsLeft) => {
        const d = Math.floor(secondsLeft / (24 * 3600));
        const h = Math.floor((secondsLeft % (24 * 3600)) / 3600);
        const m = Math.floor((secondsLeft % 3600) / 60);
        const s = secondsLeft % 60;
        
        let parts = [];
        if (d > 0) parts.push(`${d}d`);
        if (h > 0 || d > 0) parts.push(`${h}h`);
        if (m > 0 || h > 0 || d > 0) parts.push(`${m}m`);
        parts.push(`${s}s`);
        
        return `Opens in ${parts.join(' ')}`;
    };

    const formatCloseDiff = (secondsLeft) => {
        const h = Math.floor(secondsLeft / 3600);
        const m = Math.floor((secondsLeft % 3600) / 60);
        const s = secondsLeft % 60;
        return `Closes in ${h}h ${m}m ${s}s`;
    };

    if (region === 'Forex' || region === 'Commodities' || region === 'FOREX' || region === 'COMMODITIES') {
        const openTimeInWeek = 0 * 24 * 3600 + 22 * 3600; // Sunday 22:00 UTC
        const closeTimeInWeek = 5 * 24 * 3600 + 22 * 3600; // Friday 22:00 UTC

        if (currentSecondsInWeek >= openTimeInWeek && currentSecondsInWeek < closeTimeInWeek) {
            return { status: 'Open', detail: formatCloseDiff(closeTimeInWeek - currentSecondsInWeek), isOpen: true };
        } else {
            let secondsUntilOpen = openTimeInWeek - currentSecondsInWeek;
            if (secondsUntilOpen < 0) secondsUntilOpen += 7 * 24 * 3600;
            return { status: 'Closed', detail: formatDiff(secondsUntilOpen), isOpen: false };
        }
    }

    let openMinDay, closeMinDay;

    switch (region) {
        case 'US': case 'USA':
            openMinDay = 14 * 60 + 30; closeMinDay = 21 * 60; break;
        case 'India': case 'IN':
            openMinDay = 3 * 60 + 45; closeMinDay = 10 * 60; break;
        case 'Europe': case 'EU':
            openMinDay = 8 * 60; closeMinDay = 16 * 60 + 30; break;
        case 'Japan': case 'JP':
            openMinDay = 0; closeMinDay = 6 * 60; break;
        case 'China': case 'CN':
            openMinDay = 1 * 60 + 30; closeMinDay = 7 * 60; break;
        case 'Singapore': case 'SG':
            openMinDay = 1 * 60; closeMinDay = 9 * 60; break;
        case 'South Korea': case 'KR':
            openMinDay = 0; closeMinDay = 6 * 60 + 30; break;
        default: return { status: 'Unknown', detail: '', isOpen: false };
    }

    const currentSecToday = utcHour * 3600 + utcMinute * 60 + utcSecond;
    const openSecDay = openMinDay * 60;
    const closeSecDay = closeMinDay * 60;
    
    const isTradingHours = currentSecToday >= openSecDay && currentSecToday < closeSecDay;

    if (utcDay >= 1 && utcDay <= 5 && isTradingHours) {
        return { status: 'Open', detail: formatCloseDiff(closeSecDay - currentSecToday), isOpen: true };
    } else {
        let nextOpenDay = utcDay;
        
        // Before open today (weekday)
        if (utcDay >= 1 && utcDay <= 5 && currentSecToday < openSecDay) {
            nextOpenDay = utcDay;
        } 
        // After close today (Mon-Thu)
        else if (utcDay >= 1 && utcDay <= 4 && currentSecToday >= closeSecDay) {
            nextOpenDay = utcDay + 1;
        } 
        // Friday after close -> Next Monday
        else if (utcDay === 5 && currentSecToday >= closeSecDay) {
            nextOpenDay = 8;
        } 
        // Saturday -> Next Monday
        else if (utcDay === 6) {
            nextOpenDay = 8;
        } 
        // Sunday -> Today (since utcDay=0, but wait, Monday is 1! So next day is 1)
        else if (utcDay === 0) {
            nextOpenDay = 1;
        } else {
            nextOpenDay = utcDay + 1;
        }

        let secondsUntilOpen = (nextOpenDay * 24 * 3600 + openSecDay) - currentSecondsInWeek;
        return { status: 'Closed', detail: formatDiff(secondsUntilOpen), isOpen: false };
    }
};
