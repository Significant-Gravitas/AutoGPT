// Content script to interact with Zoho Deluge editors
// Handles bridge between extension context and page context (Main World)

// 1. Inject the bridge script
const script = document.createElement('script');
script.src = chrome.runtime.getURL('bridge.js');
(document.head || document.documentElement).appendChild(script);

// 2. Listen for messages from the extension (IDE -> Background -> Content)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'GET_ZOHO_CODE' || request.action === 'SET_ZOHO_CODE') {
        // Relay to bridge
        window.postMessage({ type: 'FROM_EXTENSION', ...request }, '*');

        // Wait for response from bridge
        const handler = (event) => {
            if (event.data && event.data.type === 'FROM_PAGE' && event.data.action === request.action) {
                window.removeEventListener('message', handler);
                sendResponse(event.data.response);
            }
        };
        window.addEventListener('message', handler);
        return true; // Keep channel open
    }
});

// 3. Listen for unsolicited messages from bridge (e.g. console updates)
window.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'FROM_PAGE' && event.data.action === 'ZOHO_CONSOLE_UPDATE') {
        chrome.runtime.sendMessage({ action: 'ZOHO_CONSOLE_UPDATE', data: event.data.data });
    }
});
