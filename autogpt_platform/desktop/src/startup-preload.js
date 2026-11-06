"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("autogpt", {
  onRuntimeEvent: (callback) => ipcRenderer.on("runtime-event", (_, event) => callback(event)),
  onUpdateOffer: (callback) => ipcRenderer.on("update-offer", (_, offer) => callback(offer)),
  takeUpdateOffer: () => ipcRenderer.send("take-update-offer"),
  openLogs: () => ipcRenderer.send("open-logs"),
  openDoctor: () => ipcRenderer.send("open-doctor"),
  quit: () => ipcRenderer.send("quit"),
});
