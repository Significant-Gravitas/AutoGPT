"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("autogpt", {
  resetOwnerPassword: (password, confirmation) =>
    ipcRenderer.invoke("reset-owner-password", { password, confirmation }),
});
