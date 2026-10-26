"use strict";

const byId = id => document.getElementById(id);
const labels = { ok: "Ready", warning: "Needs attention", error: "Needs attention", unknown: "Not checked", stopped: "Stopped", available: "Available", configured: "Configured", unavailable: "Unavailable" };

function showTab(tab) {
  if (!["doctor", "features", "privacy", "desktop"].includes(tab)) return;
  for (const button of document.querySelectorAll("[data-tab]")) button.setAttribute("aria-pressed", String(button.dataset.tab === tab));
  for (const panel of document.querySelectorAll("main > section")) panel.hidden = panel.id !== `panel-${tab}`;
}

function notice(text) { byId("notice").textContent = text || ""; }

function renderCards(id, entries) {
  const cards = entries.map(entry => {
    const card = document.createElement("article"); card.className = "check";
    const heading = document.createElement("div"); heading.className = "check-heading";
    const title = document.createElement("h3"); title.textContent = entry.label;
    const status = document.createElement("span");
    const state = Object.hasOwn(labels, entry.status) ? entry.status : "unknown";
    status.className = `status status-${state}`; status.textContent = labels[state];
    const detail = document.createElement("p"); detail.textContent = entry.detail;
    heading.append(title, status); card.append(heading, detail);
    return card;
  });
  byId(id).replaceChildren(...cards);
}

function renderState(state) {
  byId("version").textContent = `v${state.version}`;
  for (const key of ["usageStatistics", "crashReports", "launchAtLogin", "keepRunningInBackground"]) byId(key).checked = state.preferences[key];
  byId("launchAtLogin").disabled = !state.login.supported;
  byId("login-description").textContent = state.login.reason;
  byId("login-status").textContent = state.login.status || "";
  for (const [id, key] of [["usage-destination", "usageStatistics"], ["crash-destination", "crashReports"]]) {
    const destination = state.reporting[key];
    byId(key).disabled = !destination.configured;
    if (!destination.configured) byId(key).checked = false;
    byId(id).textContent = destination.configured ? `Destination: ${destination.host}` : "Reporting is disabled in this build. Nothing is sent.";
  }
}

async function loadState() {
  try { const result = await window.doctor.state(); if (result.error) return notice(result.error); renderState(result.state); }
  catch { notice("Could not load desktop preferences."); }
}

async function runDoctor() {
  byId("run").disabled = true; byId("run").textContent = "Checking…"; notice("");
  try {
    const result = await window.doctor.run();
    if (result.error) return notice(result.error);
    byId("summary-title").textContent = result.report.summary.title;
    byId("summary-detail").textContent = result.report.summary.detail;
    renderCards("checks", result.report.checks); renderCards("capabilities", result.report.capabilities);
    byId("preview").textContent = result.preview; byId("export").disabled = false;
    notice("Checks finished. Nothing was changed. Review the report before exporting.");
  } catch { notice("Doctor could not finish its checks. Try again or open the local logs."); }
  finally { byId("run").disabled = false; byId("run").textContent = "Run Doctor"; }
}

async function save(event, keys) {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]'); button.disabled = true;
  try {
    const patch = Object.fromEntries(keys.map(key => [key, byId(key).checked]));
    const result = await window.doctor.savePreferences(patch);
    if (result.state) renderState(result.state);
    notice(result.error || "Preferences saved. Changes apply immediately.");
  } catch { notice("Could not save preferences. Try again."); }
  finally { button.disabled = false; }
}

for (const button of document.querySelectorAll("[data-tab]")) button.addEventListener("click", () => showTab(button.dataset.tab));
window.doctor.onTab(showTab);
byId("run").addEventListener("click", runDoctor);
byId("privacy-form").addEventListener("submit", event => save(event, ["usageStatistics", "crashReports"]));
byId("desktop-form").addEventListener("submit", event => save(event, ["launchAtLogin", "keepRunningInBackground"]));
byId("connections").addEventListener("click", async () => { const result = await window.doctor.openConnections(); notice(result.error); });
byId("logs").addEventListener("click", async () => { const result = await window.doctor.openLogs(); notice(result.error); });
byId("export").addEventListener("click", async () => {
  byId("export").disabled = true;
  try { const result = await window.doctor.exportReport(); notice(result.error || (result.saved ? "Report saved on this computer. Nothing was uploaded." : "Export canceled.")); }
  catch { notice("Could not export the report."); }
  finally { byId("export").disabled = false; }
});
loadState();
