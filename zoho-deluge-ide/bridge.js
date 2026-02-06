// Bridge script injected into Zoho page to access Ace/CodeMirror in Main World

(function() {
    window.addEventListener('message', (event) => {
        if (event.data && event.data.type === 'FROM_EXTENSION') {
            const action = event.data.action;
            const code = getEditorCode();

            // If this frame doesn't have an editor and we're just getting code, don't respond
            // so other frames can.
            if (action === 'GET_ZOHO_CODE' && code === null) return;

            let response = {};
            if (action === 'GET_ZOHO_CODE') {
                response = { code: code };
            } else if (action === 'SET_ZOHO_CODE') {
                response = { success: setEditorCode(event.data.code) };
            }

            window.postMessage({ type: 'FROM_PAGE', action: action, response: response }, '*');
        }
    });

    function getEditorCode() {
        // Ace Editor
        const aceEl = document.querySelector('.ace_editor');
        if (aceEl && aceEl.env && aceEl.env.editor) {
            return aceEl.env.editor.getValue();
        }

        // CodeMirror
        const cmEl = document.querySelector('.CodeMirror');
        if (cmEl && cmEl.CodeMirror) {
            return cmEl.CodeMirror.getValue();
        }

        // Fallback
        const textareas = document.querySelectorAll('textarea');
        for (let ta of textareas) {
            if (ta.value.includes('info') || ta.value.includes('zoho.')) {
                return ta.value;
            }
        }
        return null;
    }

    function setEditorCode(code) {
        const aceEl = document.querySelector('.ace_editor');
        if (aceEl && aceEl.env && aceEl.env.editor) {
            aceEl.env.editor.setValue(code);
            return true;
        }

        const cmEl = document.querySelector('.CodeMirror');
        if (cmEl && cmEl.CodeMirror) {
            cmEl.CodeMirror.setValue(code);
            return true;
        }

        const textareas = document.querySelectorAll('textarea');
        for (let ta of textareas) {
            if (ta.value.includes('info') || ta.value.includes('zoho.')) {
                ta.value = code;
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                return true;
            }
        }
        return false;
    }

    // Console scraping
    setInterval(() => {
        const consoleEl = document.querySelector('.console-output, #console-result, .builder-console-content');
        if (consoleEl && consoleEl.innerText) {
            window.postMessage({
                type: 'FROM_PAGE',
                action: 'ZOHO_CONSOLE_UPDATE',
                data: consoleEl.innerText
            }, '*');
        }
    }, 2000);
})();
