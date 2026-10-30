"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

let window = null;
const TABS = new Set(["doctor", "features", "privacy", "desktop"]);

function doctorTab(tab) {
  return TABS.has(tab) ? tab : "doctor";
}

function isDoctorSender(event, contents, url) {
  return !contents.isDestroyed() && event.sender === contents &&
    event.senderFrame === contents.mainFrame && event.senderFrame?.url === url && contents.getURL() === url;
}

function openDoctorWindow({ icon, getState, runChecks, savePreferences, openConnections, openLogs }, tab = "doctor") {
  if (window) {
    window.show();
    window.focus();
    window.webContents.send("doctor-tab", doctorTab(tab));
    return;
  }
  const { BrowserWindow, ipcMain, dialog } = require("electron");
  window = new BrowserWindow({
    width: 920, height: 780, minWidth: 620, minHeight: 550,
    title: "AutoGPT Doctor", icon, show: false, autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, "doctor-preload.js"), contextIsolation: true, sandbox: true, nodeIntegration: false },
  });
  const owned = window;
  const contents = owned.webContents;
  const file = path.join(__dirname, "doctor.html");
  const url = pathToFileURL(file).href;
  let preview = null;
  let checking = false;
  let saving = false;
  let exporting = false;
  contents.setWindowOpenHandler(() => ({ action: "deny" }));
  contents.on("will-navigate", event => event.preventDefault());
  contents.on("will-redirect", event => event.preventDefault());
  const handlers = {
    "doctor-state": async () => ({ state: await getState() }),
    "doctor-run": async () => {
      if (checking) return { error: "Doctor is already checking this installation." };
      checking = true;
      try {
        const result = await runChecks();
        preview = result.preview;
        return result;
      } finally { checking = false; }
    },
    "doctor-save-preferences": async input => {
      if (saving) return { error: "Wait for the current settings change to finish." };
      saving = true;
      try { return await savePreferences(input); }
      finally { saving = false; }
    },
    "doctor-open-connections": async () => { await openConnections(); return { ok: true }; },
    "doctor-open-logs": async () => { await openLogs(); return { ok: true }; },
    "doctor-export": async () => {
      if (!preview) return { error: "Run Doctor and inspect the report before exporting." };
      if (exporting) return { error: "A report export is already open." };
      exporting = true;
      const text = preview;
      try {
        const result = await dialog.showSaveDialog(owned, {
          title: "Export the previewed Doctor report",
          defaultPath: `AutoGPT-doctor-${new Date().toISOString().slice(0, 10)}.json`,
          filters: [{ name: "Doctor report", extensions: ["json"] }],
        });
        if (result.canceled || !result.filePath) return { canceled: true };
        const handle = await fs.open(result.filePath, "wx", 0o600);
        try { await handle.writeFile(text, "utf8"); await handle.sync(); }
        finally { await handle.close(); }
        return { saved: true };
      } catch (error) {
        return { error: error.code === "EEXIST" ? "Choose a new filename. Doctor does not replace existing files." : "The report could not be saved. Check the destination and available disk space." };
      } finally { exporting = false; }
    },
  };
  for (const [channel, handler] of Object.entries(handlers)) {
    ipcMain.handle(channel, async (event, input) => {
      if (!isDoctorSender(event, contents, url)) return { error: "Not allowed." };
      try { return await handler(input); }
      catch { return { error: "This check could not finish. Try again or open the local logs." }; }
    });
  }
  owned.on("closed", () => {
    for (const channel of Object.keys(handlers)) ipcMain.removeHandler(channel);
    if (window === owned) window = null;
  });
  contents.once("did-finish-load", () => contents.send("doctor-tab", doctorTab(tab)));
  owned.once("ready-to-show", () => owned.show());
  owned.loadFile(file);
}

module.exports = { openDoctorWindow, isDoctorSender, doctorTab };
