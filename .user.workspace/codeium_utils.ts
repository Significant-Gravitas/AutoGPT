import * as vscode from 'vscode';
import * as cheerio from 'cheerio';
import { Element } from 'domhandler';
import { Element as CheerioElement } from 'cheerio';

function activate(context: vscode.ExtensionContext) {
  let activeChatPanel: vscode.WebviewPanel | undefined;

  vscode.window.onDidChangeWindowState((e: vscode.WindowState) => {
    const activeEditor = vscode.window.activeTextEditor;

    if (!activeEditor || activeEditor.document.languageId !== 'codeium-chat') {
      activeChatPanel = undefined;
      return;
    }

    if (!activeChatPanel) {
      activeChatPanel = vscode.window.createWebviewPanel(
        'codeium-chat',
        'Codeium Chat',
        { viewColumn: activeEditor.viewColumn, preserveFocus: true },
        { enableScripts: true }
      );
    }

    const contents = activeChatPanel.webview.html;

    try {
      const $ = cheerio.load(contents);

      const messages = $('div.message').map((i: number, el: CheerioElement) => {
        const $el = $(el);
        const timestamp = $el.find('.timestamp').text();
        const message = $el.find('.message-content').text().replace(/\s+/g, ' ').trim();
        return `${timestamp} - ${message}`;
      }).get();

      // TODO: Display the extracted messages in a custom view or output channel
    } catch (error) {
      console.error('Error occurred while parsing chat panel contents:', error);
    }
  });
}

export { activate };
