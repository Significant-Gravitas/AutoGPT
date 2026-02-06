document.getElementById('openIDE').addEventListener('click', () => {
  chrome.tabs.create({ url: chrome.runtime.getURL('ide.html') });
});
