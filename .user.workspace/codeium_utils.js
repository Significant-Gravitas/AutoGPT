"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var vscode = require("vscode");
var cheerio = require("cheerio");
function activate(context) {
    var activeChatPanel;
    vscode.window.onDidChangeWindowState(function (e) {
        var activeEditor = vscode.window.activeTextEditor;
        if (!activeEditor || activeEditor.document.languageId !== 'codeium-chat') {
            activeChatPanel = undefined;
            return;
        }
        if (!activeChatPanel) {
            activeChatPanel = vscode.window.createWebviewPanel('codeium-chat', 'Codeium Chat', { viewColumn: activeEditor.viewColumn, preserveFocus: true }, { enableScripts: true });
        }
        var contents = activeChatPanel.webview.html;
        try {
            var $_1 = cheerio.load(contents);
            var messages = $_1('div.message').map(function (i, el) {
                var $el = $_1(el);
                var timestamp = $el.find('.timestamp').text();
                var message = $el.find('.message-content').text().replace(/\s+/g, ' ').trim();
                return "".concat(timestamp, " - ").concat(message);
            }).get();
            // TODO: Display the extracted messages in a custom view or output channel
        }
        catch (error) {
            console.error('Error occurred while parsing chat panel contents:', error);
        }
    });
}
