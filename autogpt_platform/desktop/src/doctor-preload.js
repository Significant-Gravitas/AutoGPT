"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("doctor", {
  state: () => ipcRenderer.invoke("doctor-state"),
  run: () => ipcRenderer.invoke("doctor-run"),
  savePreferences: patch => ipcRenderer.invoke("doctor-save-preferences", patch),
  openConnections: () => ipcRenderer.invoke("doctor-open-connections"),
  openLogs: () => ipcRenderer.invoke("doctor-open-logs"),
  exportReport: () => ipcRenderer.invoke("doctor-export"),
  onTab: callback => ipcRenderer.on("doctor-tab", (_, tab) => callback(tab)),
});
