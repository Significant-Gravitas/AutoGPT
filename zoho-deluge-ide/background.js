// Background script for Zoho Deluge IDE

let lastZohoTabId = null;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'GET_ZOHO_CODE') {
        findZohoTab((tab) => {
            if (tab) {
                lastZohoTabId = tab.id;
                chrome.tabs.sendMessage(tab.id, { action: 'GET_ZOHO_CODE' }, (response) => {
                    sendResponse(response);
                });
            } else {
                sendResponse({ error: 'No Zoho Deluge tab found' });
            }
        });
        return true; // Keep channel open
    }

    if (request.action === 'SET_ZOHO_CODE') {
        if (lastZohoTabId) {
            chrome.tabs.sendMessage(lastZohoTabId, { action: 'SET_ZOHO_CODE', code: request.code }, (response) => {
                sendResponse(response);
            });
        } else {
            findZohoTab((tab) => {
                if (tab) {
                    lastZohoTabId = tab.id;
                    chrome.tabs.sendMessage(tab.id, { action: 'SET_ZOHO_CODE', code: request.code }, (response) => {
                        sendResponse(response);
                    });
                } else {
                    sendResponse({ error: 'No Zoho tab connected' });
                }
            });
        }
        return true;
    }

    if (request.action === 'ZOHO_CONSOLE_UPDATE') {
        // Forward console update to IDE tab
        chrome.runtime.sendMessage({ action: 'IDE_CONSOLE_UPDATE', data: request.data });
    }
});

function findZohoTab(callback) {
    chrome.tabs.query({ url: '*://*.zoho.com/*' }, (tabs) => {
        // Try to find a tab that likely has an editor
        const editorTab = tabs.find(t =>
            t.url.includes('creator') ||
            t.url.includes('crm') ||
            t.url.includes('workflow') ||
            t.url.includes('builder')
        );
        callback(editorTab || tabs[0]);
    });
}
