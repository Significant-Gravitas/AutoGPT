let editor;

// Since ide.js is now injected after monaco is ready, we can just init
initEditor();

function initEditor() {
    if (typeof registerDelugeLanguage === 'function') {
        registerDelugeLanguage();
    }

    editor = monaco.editor.create(document.getElementById('editor-container'), {
        value: '// Start coding in Zoho Deluge...\n\ninfo "Hello, World!";',
        language: 'deluge',
        theme: 'vs-dark',
        automaticLayout: true,
        fontSize: 14,
        minimap: { enabled: true },
        scrollBeyondLastLine: false,
        lineNumbers: 'on',
        roundedSelection: false,
        cursorStyle: 'line',
        glyphMargin: true,
        readOnly: false // Explicitly ensure it is NOT read-only
    });

    // Update cursor position in status bar
    editor.onDidChangeCursorPosition((e) => {
        document.getElementById('cursor-pos').innerText = `Ln ${e.position.lineNumber}, Col ${e.position.column}`;
    });

    // Load last saved code
    chrome.storage.local.get(['saved_deluge_code'], (result) => {
        if (result.saved_deluge_code) {
            editor.setValue(result.saved_deluge_code);
            log('System', 'Loaded previously saved code.');
        }
    });

    log('System', 'Monaco Editor initialized successfully. Mode: Local');

    // Auto-save on change
    let autoSaveTimeout;
    editor.onDidChangeModelContent(() => {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(() => {
            const code = editor.getValue();
            chrome.storage.local.set({ 'saved_deluge_code': code });
            document.getElementById('sync-status').innerText = 'Auto-saved';
            setTimeout(() => {
                const syncStatusEl = document.getElementById('sync-status');
                if (syncStatusEl.innerText === 'Auto-saved') {
                    checkConnection(); // Restore connection status
                }
            }, 2000);
        }, 1000);
    });

    setupEventHandlers();
    checkConnection(); // Initial connection check
    setInterval(checkConnection, 5000); // Heartbeat
}

function checkConnection() {
    chrome.runtime.sendMessage({ action: 'CHECK_CONNECTION' }, (response) => {
        const statusEl = document.getElementById('status-indicator');
        const syncStatusEl = document.getElementById('sync-status');

        if (response && response.connected) {
            statusEl.innerText = 'Connected: ' + (response.tabTitle || 'Zoho Tab');
            statusEl.style.color = '#4ec9b0';
            syncStatusEl.innerText = 'Connected';
        } else {
            statusEl.innerText = 'Disconnected (Local Mode)';
            statusEl.style.color = '#888';
            syncStatusEl.innerText = 'Local';
        }
    });
}

function setupEventHandlers() {
    document.getElementById('new-btn').addEventListener('click', () => {
        if (confirm('Start a new script? Unsaved changes in the editor will be lost.')) {
            editor.setValue('// New Zoho Deluge Script\n\n');
        }
    });
    document.getElementById('pull-btn').addEventListener('click', pullFromZoho);
    document.getElementById('push-btn').addEventListener('click', pushToZoho);
    document.getElementById('save-btn').addEventListener('click', saveLocally);
    document.getElementById('clear-console').addEventListener('click', () => {
        const activePanel = document.querySelector('.panel-content.active');
        if (activePanel) activePanel.innerHTML = '';
    });

    document.getElementById('manual-console-btn').addEventListener('click', () => {
        const result = prompt('Enter manual console output:');
        if (result) updateConsole(result);
    });

    // Panel tab switching
    document.querySelectorAll('.panel-header .tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.panel-header .tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel-content').forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const targetId = tab.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            saveLocally();
        }
    });

    // Listen for console updates from background
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === 'IDE_CONSOLE_UPDATE') {
            updateConsole(request.data);
        }
    });
}

function updateConsole(data) {
    const consoleOutput = document.getElementById('console-output');
    // Simple deduplication or just append if changed
    if (consoleOutput.dataset.lastOutput !== data) {
        log('Zoho', data);
        consoleOutput.dataset.lastOutput = data;
    }
}

function log(type, message) {
    const consoleOutput = document.getElementById('console-output');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type.toLowerCase()}`;
    const timestamp = new Date().toLocaleTimeString();
    entry.innerText = `[${timestamp}] ${type.toUpperCase()}: ${message}`;
    consoleOutput.appendChild(entry);
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function pullFromZoho() {
    log('System', 'Searching for active Zoho Deluge editor...');
    chrome.runtime.sendMessage({ action: 'GET_ZOHO_CODE' }, (response) => {
        if (response && response.code) {
            editor.setValue(response.code);
            log('Success', 'Code pulled from Zoho tab.');
        } else {
            log('Error', 'No Zoho Deluge editor found or tab not connected.');
        }
    });
}

function pushToZoho() {
    const code = editor.getValue();
    log('System', 'Pushing code to Zoho tab...');
    chrome.runtime.sendMessage({ action: 'SET_ZOHO_CODE', code: code }, (response) => {
        if (response && response.success) {
            log('Success', 'Code pushed to Zoho tab.');
        } else {
            log('Error', 'Failed to push code. Ensure the Zoho tab is open and active.');
        }
    });
}

function saveLocally() {
    const code = editor.getValue();
    chrome.storage.local.set({ 'saved_deluge_code': code }, () => {
        log('Success', 'Code saved locally.');
    });
}
