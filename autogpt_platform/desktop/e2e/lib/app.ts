// Starting and quitting the installed app, and the checks that go with a
// quit: nothing left running, nothing changed in the installed tree, nothing
// the firewall noticed.

import fs from "node:fs";
import path from "node:path";

import { expect, test, type Browser, type Page } from "@playwright/test";

import { attach, freePort } from "./attach";
import { recordDuration } from "./durations";
import { dataDir, dataDirOverride } from "./config";
import { firewallReport } from "./firewall";
import { appProcesses, platform } from "./platform";
import { describe, isAlive, recordedPids } from "./processes";
import { logPosition, logTails, waitForIdle, waitForReady } from "./runtime-log";
import { changedPaths, summarize, takeSnapshot, type Snapshot } from "./snapshot";

// The shell gives its runtime this long to stop and then kills it, with
// every service it recorded (src/runtime.js STOP_GRACE_MS). A healthy stop
// takes a few seconds and 45 at worst.
const SHELL_STOP_GRACE_MS = 60_000;
const QUIT_TIMEOUT_MS = SHELL_STOP_GRACE_MS + 15_000;
// The runtime counts what is in progress every two seconds, and a run that
// lost its lock is asked about every ten (runtime/autogpt_desktop/runs.py).
const IDLE_TIMEOUT_MS = 30_000;
const ATTACH_TIMEOUT_MS = 60_000;

export interface RunningApp {
  url: string;
  browser: Browser;
  page: Page;
  /** The installed tree as it was before this start; null for an AppImage. */
  installed: Snapshot | null;
  /** Where this start's lines begin in runtime.log. */
  logStart: number;
}

/** What the launcher and the shell printed on the latest start. */
let launcherLog: string | null = null;

/** `what` names the start in the durations this run records (lib/durations.ts). */
export async function startApp(readyTimeoutMs: number, what = "restart"): Promise<RunningApp> {
  const running = appProcesses();
  if (running.length > 0) {
    throw new Error(`the app is already running; only one instance can run:\n${describe(running)}`);
  }
  const data = dataDir();
  const installDir = platform.installDir();
  const installed = installDir ? takeSnapshot(installDir) : null;
  const position = logPosition(data);
  const debugPort = await freePort();
  const started = Date.now();
  launcherLog = path.join(test.info().project.outputDir, "launcher", `start-${started}.log`);
  await platform.launch([`--remote-debugging-port=${debugPort}`, ...profileSwitch()], launcherLog);
  const url = await waitForReady(data, position, readyTimeoutMs, {
    isRunning: () => appProcesses().length > 0,
    output: launcherOutput,
  });
  recordDuration(`${what}, to ready`, (Date.now() - started) / 1000);
  const { browser, page } = await attach(debugPort, url, ATTACH_TIMEOUT_MS);
  // Kept only if a test fails (recordFailure). Not worth failing a test over.
  await page.context().tracing.start({ screenshots: true, snapshots: true }).catch(() => undefined);
  return { url, browser, page, installed, logStart: position };
}

/** For afterEach: keep the logs, the window and the trace of a failed test. */
export async function recordFailure(app: RunningApp | null): Promise<void> {
  const info = test.info();
  if (info.status === info.expectedStatus) return;
  await info.attach("logs", { body: logTails(dataDir()), contentType: "text/plain" });
  if (launcherLog) await info.attach("launcher", { body: launcherOutput(), contentType: "text/plain" });
  if (!app) return;
  const window = await app.page.screenshot().catch(() => null);
  if (window) await info.attach("window", { body: window, contentType: "image/png" });
  const trace = info.outputPath("trace.zip");
  const saved = await app.page.context().tracing.stop({ path: trace }).then(
    () => true,
    () => false,
  );
  if (saved) await info.attach("trace", { path: trace, contentType: "application/zip" });
}

/** Quit, then check what the whole time since `startApp` left behind. */
export async function quitAndVerify(app: RunningApp): Promise<void> {
  await quit(app);
  await verifyInstallUntouched(app.installed);
  await verifyFirewall();
}

/** Quit as a user does, by closing the main window, and check that the app
 * stopped by itself and left nothing running.
 *
 * While the shell takes an agent run or an AutoPilot turn to be in progress,
 * closing the window asks what to do with it (src/quit.js), in a native
 * dialog nothing here can answer. What the shell knows is what the runtime
 * last told it, which is a few seconds old: a run that a test has just seen
 * finish may still be counted. So the close waits until the shell has been
 * told that nothing is running, and a test that leaves something running
 * fails here, saying so, instead of waiting out a question nobody answers. */
export async function quit(app: RunningApp): Promise<void> {
  const data = dataDir();
  const services = recordedPids(data);
  expect(services.length, "the runtime records the services it started").toBeGreaterThan(0);
  const counted = await waitForIdle(data, app.logStart, IDLE_TIMEOUT_MS);
  expect(counted, "agent runs and AutoPilot turns still in progress: closing the window would ask about them").toBe(0);

  await app.page.context().tracing.stop().catch(() => undefined);
  // Closing the window ends the debugging session too, so the call may not
  // get its answer.
  await app.page.close().catch(() => undefined);
  const started = Date.now();
  await waitForExit(services, QUIT_TIMEOUT_MS);
  const took = Date.now() - started;
  note("seconds to quit", String(Math.round(took / 1000)));
  await app.browser.close().catch(() => undefined);

  const left = appProcesses();
  expect.soft(describe(left), "processes still running after quitting").toBe("");
  expect.soft(services.filter(isAlive), "recorded services still alive").toEqual([]);
  expect.soft(fs.existsSync(path.join(data, "run", "children.json")), "run/children.json is removed").toBe(false);
  // None of the three above tells a stop from a kill: the shell kills a
  // runtime that hangs, then the services it recorded, and removes the list
  // itself. These two do.
  expect.soft(took, "the runtime stopped by itself, before the shell would kill it").toBeLessThan(SHELL_STOP_GRACE_MS);
  expect.soft(postgresLockFile(data), "PostgreSQL was shut down, not killed").toBeNull();
}

export async function verifyInstallUntouched(before: Snapshot | null): Promise<void> {
  const installDir = platform.installDir();
  if (before && installDir) {
    const changed = changedPaths(before, takeSnapshot(installDir));
    expect.soft(changed.length === 0 ? "" : summarize(changed), `running the app changed ${installDir}`).toBe("");
  }
  expect.soft(await platform.damage(), "the installed app still passes its integrity check").toBeNull();
}

export async function verifyFirewall(): Promise<void> {
  const report = await firewallReport();
  if (!report) return;
  for (const line of report.notes) note("firewall", line);
  expect.soft(report.violations, "sockets the Windows firewall would ask about").toEqual([]);
}

/** Report leftovers without treating a shared executable path as proof of ownership. */
export async function stopStrayApp(): Promise<void> {
  const left = appProcesses();
  if (left.length === 0) return;
  console.log(`processes still using the installed executables:\n${describe(left)}\n${logTails(dataDir())}`);
  throw new Error(
    "Automatic cleanup refused: executable paths do not prove test ownership. " +
    "No processes were terminated. Verify the reported PIDs and perform manual cleanup of owned test processes before rerunning.",
  );
}

export async function waitForExit(pids: number[], timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!pids.some(isAlive) && appProcesses().length === 0) return;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/** PostgreSQL removes postmaster.pid when it shuts down and leaves it when
 * it is killed. Returns the file if it is there. */
export function postgresLockFile(data: string): string | null {
  const file = path.join(data, "postgres", "postmaster.pid");
  return fs.existsSync(file) ? file : null;
}

export function note(type: string, description: string): void {
  test.info().annotations.push({ type, description });
  console.log(`${type}: ${description}`);
}

/** A data directory of the tests' own gets a browser profile of its own.
 * The shell keeps cookies in Electron's profile, which is not under the data
 * directory, and a second data directory is served from the same address: on
 * the default profile the tests would replace the session of the install the
 * developer uses. */
function profileSwitch(): string[] {
  return dataDirOverride ? [`--user-data-dir=${path.join(dataDir(), "electron-profile")}`] : [];
}

function launcherOutput(): string {
  try {
    return (launcherLog && fs.readFileSync(launcherLog, "utf8").trim()) || "(nothing)";
  } catch {
    return "(nothing)";
  }
}
