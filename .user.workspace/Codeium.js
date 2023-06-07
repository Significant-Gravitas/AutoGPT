import * as vscode from 'vscode';
import * as cheerio from 'cheerio';

export function activate(context: vscode.ExtensionContext) {
let activeChatPanel: vscode.WebviewPanel | undefined = undefined;

// Register a handler for the onDidChangeWindowState event
vscode.window.onDidChangeWindowState((e) => {
 // Check if the chat panel is focused
 if (
   e.focusedWindow &&
   e.focusedWindow.title.includes('Chat') &&
   e.focusedWindow.title.includes('Codeium')
 ) {
   // Get the webview panel associated with the chat panel
   const chatPanel = vscode.window.activeWebviewPanel;
   if (chatPanel) {
     // Store a reference to the active chat panel
     activeChatPanel = chatPanel;

     // Get the contents of the chat panel
     const contents = chatPanel.webview.html;

     // Parse the HTML source using Cheerio
     const $ = cheerio.load(contents);

     // Extract the relevant information from the parsed HTML
     const messages = $('div.message')
       .map((i, el) => {
         const $el = $(el);
         const timestamp = $el.find('.timestamp').text();
         const message = $el
           .find('.message-content')
           .text()
           .replace(/\s+/g, ' ')
           .trim();
         return `${timestamp} - ${message}`;
       })
       .get();

     // TODO: Display the extracted messages in a custom view or output channel
   }
 } else {
   // Clear the reference to the active chat panel
   activeChatPanel = undefined;
 }
});
}