// Background script for Zoho Deluge IDE

let lastZohoTabId = null;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'CHECK_CONNECTION') {
        findZohoTab((tab) => {
            if (tab) {
                sendResponse({ connected: true, tabTitle: tab.title });
            } else {
                sendResponse({ connected: false });
            }
        });
        return true;
    }

    if (request.action === 'GET_ZOHO_CODE') {
        findZohoTab((tab) => {
            if (tab) {
                lastZohoTabId = tab.id;
                chrome.tabs.sendMessage(tab.id, { action: 'GET_ZOHO_CODE' }, (response) => {
                    if (chrome.runtime.lastError) {
                        sendResponse({ error: 'Tab not responding' });
                    } else {
                        sendResponse(response);
                    }
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
    // Search across all Zoho domains and include incognito
    chrome.tabs.query({}, (tabs) => {
        const zohoTabs = tabs.filter(t =>
            t.url && (
                t.url.includes('zoho.com') ||
                t.url.includes('zoho.eu') ||
                t.url.includes('zoho.in') ||
                t.url.includes('zoho.com.au') ||
                t.url.includes('zoho.jp')
            )
        );

        // Prioritize tabs that look like editors
        const editorTab = zohoTabs.find(t =>
            t.url.includes('creator') ||
            t.url.includes('crm') ||
            t.url.includes('workflow') ||
            t.url.includes('builder') ||
            t.title.toLowerCase().includes('deluge') ||
            t.title.toLowerCase().includes('editor')
        );

        callback(editorTab || zohoTabs[0]);
    });
}
