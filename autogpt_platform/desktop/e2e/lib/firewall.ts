// Runs firewall.ps1, which explains what is checked and why. Windows only,
// and only when AUTOGPT_E2E_FIREWALL=1: starting the check changes the
// machine's firewall and audit settings.

import fs from "node:fs";
import path from "node:path";

import { checkFirewall } from "./config";
import { platform, run } from "./platform";
import { STATE_DIR } from "./state";

const SCRIPT = path.join(__dirname, "firewall.ps1");
const STATE = path.join(STATE_DIR, "firewall.json");

export interface FirewallReport {
  violations: string[];
  notes: string[];
}

/** Start recording, once per run; every later check covers everything since. */
export async function beginFirewallCheck(): Promise<string | null> {
  if (!checkFirewall) return null;
  fs.mkdirSync(STATE_DIR, { recursive: true });
  const result = await powershell("begin", ["-NodePath", process.execPath]);
  return result.trim();
}

export async function firewallReport(): Promise<FirewallReport | null> {
  if (!checkFirewall || !fs.existsSync(STATE)) return null;
  const output = await powershell("check", []);
  const report = JSON.parse(output.slice(output.indexOf("{")));
  return { violations: report.violations || [], notes: report.notes || [] };
}

async function powershell(phase: string, extra: string[]): Promise<string> {
  const installDir = platform.installDir();
  if (!installDir) throw new Error("the firewall check needs an install directory");
  const result = await run("powershell.exe", [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    SCRIPT,
    "-Phase",
    phase,
    "-InstallDir",
    installDir,
    "-StatePath",
    STATE,
    ...extra,
  ]);
  return result.output;
}
