export const getLogoDataUrl = async () => {
    try {
        const res = await fetch('/logo.png');
        if (!res.ok) throw new Error("Logo fetch failed");
        const blob = await res.blob();
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
        });
    } catch (error) {
        console.error("Error loading logo for PDF:", error);
        return null;
    }
};

export const addInteractiveLogoToPage = (doc, pageWidth, logoDataUrl) => {
    const x = 14;
    const y = 8;
    // The source PNG file is identically a square (1024x1024). 
    // We enforce a consistent 1:1 scale mapping here to prevent any squishing.
    const size = 22; // Greatly increased over the initial 12. Leaves 2-unit margin before y=32 headers.

    // Add Logo Image at the left margin without stretching it
    if (logoDataUrl) {
        doc.addImage(logoDataUrl, 'PNG', x, y, size, size);
    }

    // Interactive link overlay over the image bounding box
    doc.link(x, y, size, size, { url: window.location.origin });
};
