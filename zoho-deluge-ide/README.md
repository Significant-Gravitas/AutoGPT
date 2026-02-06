# Zoho Deluge Advanced IDE (Chrome Extension)

This extension provides a powerful, full-screen development environment for Zoho Deluge scripts, directly in your browser.

## Features

- **Monaco Editor**: The same engine that powers VS Code.
- **Zoho Autocomplete**: Intelligent suggestions for CRM, Creator, Books, Analytics, Recruit, Inventory, and Bigin.
- **Real-time Sync**: Pull code from an active Zoho editor tab and push your changes back instantly.
- **Console Integration**: Scrapes results from the Zoho console and displays them in the IDE.
- **Custom Themes**: Dark-themed, developer-friendly UI.

## Installation

1. Download or clone this repository.
2. Open Google Chrome and navigate to `chrome://extensions/`.
3. Enable **Developer mode** (toggle in the top right corner).
4. Click the **Load unpacked** button.
5. Select the `zoho-deluge-ide` folder from this repository.

## How to Use

1. **Open the IDE**: Click the extension icon in your Chrome toolbar and click "Open Full IDE".
2. **Connect to Zoho**: Ensure you have a Zoho Deluge editor open in another tab (e.g., a CRM Workflow or Creator Function).
3. **Pull Code**: Click the **Pull Code** button in the IDE top bar. The IDE will find your active Zoho editor and import the code.
4. **Develop**: Use the advanced editor with full Deluge syntax highlighting and autocomplete.
5. **Sync Back**: When ready, click **Sync to Zoho**. This will push your code back to the Zoho tab's editor.
6. **Save**: Use `Ctrl+S` (or `Cmd+S`) to save your code to the extension's local storage.

## Troubleshooting

- **Sync not working**: Ensure the Zoho tab is fully loaded and you are on a page that contains a Deluge editor (Ace or CodeMirror).
- **Console not updating**: Some Zoho products may have different console layouts; use the "manual input" button (➕) in the console panel if scraping fails.
