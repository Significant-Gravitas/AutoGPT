"use strict";

// The "Reset owner password…" window. The page only collects two inputs;
// checking them and writing the file happen here, in the main process.

const path = require("node:path");
const { BrowserWindow, ipcMain } = require("electron");

const { passwordProblem, writePasswordReset } = require("./owner");

let window = null;

// `onWritten` runs once the one-shot file is in place: restart the runtime.
function openResetPasswordWindow({ icon, dataDir, onWritten }) {
  if (window) return window.focus();
  window = new BrowserWindow({
    width: 440,
    height: 400,
    resizable: false,
    minimizable: false,
    maximizable: false,
    title: "Reset owner password",
    icon,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "reset-password-preload.js"),
      contextIsolation: true,
      sandbox: true,
    },
  });
  const contents = window.webContents;
  // The page has nowhere to go and nothing to open.
  contents.on("will-navigate", (event) => event.preventDefault());
  ipcMain.handle("reset-owner-password", (event, input) =>
    event.sender === contents ? submit(input, dataDir, onWritten) : { error: "Not allowed." },
  );
  window.loadFile(path.join(__dirname, "reset-password.html"));
  window.once("ready-to-show", () => window?.show());
  window.on("closed", () => {
    ipcMain.removeHandler("reset-owner-password");
    window = null;
  });
}

function submit(input, dataDir, onWritten) {
  const { password, confirmation } = input || {};
  const problem = passwordProblem(password, confirmation);
  if (problem) return { error: problem };
  try {
    writePasswordReset(dataDir, password);
  } catch (error) {
    return { error: `Could not save the new password: ${error.message}` };
  }
  // Answer the page first; the restart closes this window with the others.
  setImmediate(() => {
    window?.destroy();
    onWritten();
  });
  return { error: null };
}

module.exports = { openResetPasswordWindow };
