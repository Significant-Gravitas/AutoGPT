// A machine that has never seen AutoGPT: install it, start it, create the
// first account, run an agent, quit.

import { randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

import { createCalculatorAgent, executionResult, runAndWatch, type Graph } from "../lib/agent";
import { note, quitAndVerify, recordFailure, startApp, stopStrayApp, type RunningApp } from "../lib/app";
import { dataDir, firstReadyTimeoutMs, installer, kind, product } from "../lib/config";
import { timed } from "../lib/durations";
import { beginFirewallCheck } from "../lib/firewall";
import { appProcesses, platform } from "../lib/platform";
import { describe } from "../lib/processes";
import { appVersion, call, completeOnboarding, currentUser, signUp } from "../lib/session";
import { resetState, saveState } from "../lib/state";

// `ready` means the API and the frontend answer; the executor may still be
// starting (runtime/autogpt_desktop/supervisor.py), so the run gets its own wait.
const RUN_TIMEOUT_MS = 180_000;

test.describe.configure({ mode: "serial" });

let app: RunningApp | null = null;
let graph: Graph;
const email = `owner-${randomBytes(4).toString("hex")}@example.com`;
const password = randomBytes(12).toString("base64url");
const agentName = `Desktop check ${randomBytes(3).toString("hex")}`;

test.afterEach(() => recordFailure(app));
test.afterAll(stopStrayApp);

test("installs without starting anything", async () => {
  test.setTimeout(45 * 60_000);
  resetState();
  const firewall = await beginFirewallCheck();
  if (firewall) note("firewall", firewall);

  if (installer) {
    expect(platform.isInstalled(), "the app is already installed; this test needs a machine without it").toBe(false);
    const notes = await timed("install", () => platform.install(installer!));
    for (const [type, description] of Object.entries(notes)) note(type, description);
  }
  expect(platform.isInstalled(), `nothing installed at ${platform.executable()}`).toBe(true);
  expect(describe(appProcesses()), "a silent install must not start the app").toBe("");
});

test("starts for the first time", async () => {
  expect(
    fs.existsSync(path.join(dataDir(), "postgres")),
    `${dataDir()} already holds a database; the first run needs an empty data directory`,
  ).toBe(false);

  app = await startApp(firstReadyTimeoutMs, "first start");
  await app.page.context().clearCookies();
  saveState({ url: app.url, version: await appVersion(app.page) });
  checkLinuxSandbox();
});

test("the first account is the owner, and registration then closes", async () => {
  const { page, url } = app!;
  await signUp(page, url, email, password);
  saveState({ email, password });
  expect((await completeOnboarding(page)).status).toBe(200);

  // The role is in the very first session: no signing out and in again.
  expect(await currentUser(page)).toMatchObject({ email, role: "admin" });

  // The frontend sends anyone who is not an admin from /admin to /.
  await page.goto(`${url}/admin/marketplace`);
  expect(new URL(page.url()).pathname).toBe("/admin/marketplace");

  // The backend checks the role claim of the token on its own
  // (backend/api/features/admin/execution_analytics_routes.py).
  const adminOnly = await call(page, "GET", "/api/proxy/api/executions/admin/execution_analytics/config");
  expect(adminOnly.status, JSON.stringify(adminOnly.body)).toBe(200);

  // It is one person's computer: once it has its owner, nobody else can
  // create an account unless the owner allows it.
  const second = await call(page, "POST", "/api/auth/sign-up/email", {
    email: `second-${randomBytes(4).toString("hex")}@example.com`,
    password: randomBytes(12).toString("base64url"),
    name: "second",
  });
  expect(second.status, JSON.stringify(second.body)).toBeGreaterThanOrEqual(400);
  expect(await currentUser(page)).toMatchObject({ email, role: "admin" });
});

test("builds an agent and runs it", async () => {
  const { page, url } = app!;
  const created = await createCalculatorAgent(page, agentName);
  expect(created.status, JSON.stringify(created.body)).toBe(200);
  graph = { id: created.body.id, version: created.body.version };

  const run = await runAndWatch(page, graph, RUN_TIMEOUT_MS);
  expect(run.outcome, JSON.stringify(run.messages)).toBe("COMPLETED");

  const result = await executionResult(page, graph.id, run.executionId!);
  expect(result.status).toBe(200);
  expect(result.body.status).toBe("COMPLETED");
  const outputs = result.body.node_executions.map((node: any) => node.output_data?.result);
  expect(outputs).toEqual([[15]]);
  saveState({ agentName, graphId: graph.id, executionId: run.executionId! });

  await page.goto(`${url}/library`);
  await expect(page.getByText(agentName).first()).toBeVisible({ timeout: 60_000 });
});

test("quits and leaves nothing behind", async () => {
  const running = app!;
  app = null;
  await quitAndVerify(running);
});

/** The .deb keeps Chromium's sandbox through the AppArmor profile it
 * installs; the AppImage's launcher drops the sandbox where unprivileged
 * user namespaces are restricted (Ubuntu 23.10 and later). */
function checkLinuxSandbox(): void {
  if (process.platform !== "linux") return;
  const main = appProcesses().filter((found) => !found.command.includes("--type="));
  const unsandboxed = main.some((found) => found.command.includes("--no-sandbox"));
  const restricted = readFlag("/proc/sys/kernel/apparmor_restrict_unprivileged_userns") === "1";
  note("chromium sandbox", `${unsandboxed ? "off" : "on"} (user namespaces restricted: ${restricted})`);
  if (kind === "deb") {
    if (restricted) expect(fs.existsSync(`/etc/apparmor.d/${product.packageName}`), "the .deb installs an AppArmor profile").toBe(true);
    expect(unsandboxed, "the .deb runs with Chromium's sandbox").toBe(false);
  } else if (!restricted) {
    expect(unsandboxed, "nothing here stops the AppImage using Chromium's sandbox").toBe(false);
  }
}

function readFlag(file: string): string {
  try {
    return fs.readFileSync(file, "utf8").trim();
  } catch {
    return "";
  }
}
