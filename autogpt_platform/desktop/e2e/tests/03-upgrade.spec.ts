// Install a newer build over the one in use. The data must survive, and on
// Windows the installer has to cope with the app still running, because that
// is how people run installers there.

import { expect, test } from "@playwright/test";

import { executionResult } from "../lib/agent";
import {
  note,
  postgresLockFile,
  quitAndVerify,
  recordFailure,
  startApp,
  stopStrayApp,
  waitForExit,
  type RunningApp,
} from "../lib/app";
import { dataDir, readyTimeoutMs, upgradeInstaller } from "../lib/config";
import { timed } from "../lib/durations";
import { appProcesses, platform } from "../lib/platform";
import { describe, recordedPids } from "../lib/processes";
import { appVersion, currentUser, isNewer, signIn, signOut } from "../lib/session";
import { loadState } from "../lib/state";

// The installer closes the window; the runtime then notices and stops its
// services, which the shell allows a minute for.
const STOP_AFTER_INSTALLER_MS = 120_000;

test.describe.configure({ mode: "serial" });
test.skip(!upgradeInstaller, "no newer build given (AUTOGPT_E2E_UPGRADE_INSTALLER)");

let app: RunningApp | null = null;

test.afterEach(() => recordFailure(app));
test.afterAll(stopStrayApp);

test("installs a newer version over the old one", async () => {
  test.setTimeout(60 * 60_000);
  // Nothing to upgrade unless the first run got as far as an account and an agent.
  loadState();
  let services: number[] = [];
  if (process.platform === "win32") {
    app = await startApp(readyTimeoutMs);
    services = recordedPids(dataDir());
    note("upgrade", "installed while the app was running");
  }

  const notes = await timed("upgrade (the installer, over the old version)", () => platform.install(upgradeInstaller!));
  for (const [type, description] of Object.entries(notes)) note(type, description);

  if (app) {
    await app.browser.close().catch(() => undefined);
    app = null;
    await waitForExit(services, STOP_AFTER_INSTALLER_MS);
    expect(describe(appProcesses()), "the old version's processes are gone after the upgrade").toBe("");
    // The installer ends the app its own way; the next test shows whether
    // the data survived that.
    note("PostgreSQL during the upgrade", postgresLockFile(dataDir()) ? "killed" : "shut down");
  }
  expect(platform.isInstalled()).toBe(true);
});

test("the new version starts with the old data", async () => {
  const state = loadState();
  app = await startApp(readyTimeoutMs, "first start after the upgrade");
  const { page, url } = app;
  expect(app.url).toBe(state.url);

  const version = await appVersion(page);
  expect(isNewer(version, state.version), `running ${version} after upgrading from ${state.version}`).toBe(true);

  if (await currentUser(page)) await signOut(page, url);
  await signIn(page, url, state.email, state.password);
  expect(await currentUser(page)).toMatchObject({ email: state.email, role: "admin" });
  const result = await executionResult(page, state.graphId, state.executionId);
  expect(result.status, JSON.stringify(result.body)).toBe(200);
  expect(result.body.status).toBe("COMPLETED");
  await page.goto(`${url}/library`);
  await expect(page.getByText(state.agentName).first()).toBeVisible({ timeout: 60_000 });
});

test("quits and leaves nothing behind", async () => {
  const running = app!;
  app = null;
  await quitAndVerify(running);
});
