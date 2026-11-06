// The second start: same address, same account, same data. Then the case of
// an install that predates the owner step, which has an account but no admin.

import { randomBytes } from "node:crypto";

import { expect, test } from "@playwright/test";

import { executionResult } from "../lib/agent";
import { note, quit, quitAndVerify, recordFailure, startApp, stopStrayApp, type RunningApp } from "../lib/app";
import { readyTimeoutMs } from "../lib/config";
import { executeSql } from "../lib/database";
import { call, currentUser, signIn, signOut } from "../lib/session";
import { loadState } from "../lib/state";

test.describe.configure({ mode: "serial" });

let app: RunningApp | null = null;

test.afterEach(() => recordFailure(app));
test.afterAll(stopStrayApp);

test("starts again at the same address", async () => {
  const state = loadState();
  app = await startApp(readyTimeoutMs);
  // Sessions and OAuth redirect URIs are tied to the origin.
  expect(app.url).toBe(state.url);
});

test("the account and its data are still there", async () => {
  const state = loadState();
  const { page, url } = app!;
  const remembered = await currentUser(page);
  note("session kept across the restart", remembered ? "yes" : "no");
  if (remembered) await signOut(page, url);

  await signIn(page, url, state.email, state.password);
  expect(await currentUser(page)).toMatchObject({ email: state.email, role: "admin" });

  const result = await executionResult(page, state.graphId, state.executionId);
  expect(result.status, JSON.stringify(result.body)).toBe(200);
  expect(result.body.status).toBe("COMPLETED");
  await page.goto(`${url}/library`);
  await expect(page.getByText(state.agentName).first()).toBeVisible({ timeout: 60_000 });
});

test("registration is closed, and says so", async () => {
  const { page } = app!;
  const state = loadState();
  const attempt = await call(page, "POST", "/api/auth/sign-up/email", {
    email: `second-${randomBytes(4).toString("hex")}@example.com`,
    password: randomBytes(12).toString("base64url"),
    name: "second",
  });
  expect(attempt.status, JSON.stringify(attempt.body)).toBeGreaterThanOrEqual(400);
  // The wording the sign-up page turns into its "not allowed" dialog
  // (frontend/src/lib/auth/signup-gate.ts).
  expect(JSON.stringify(attempt.body)).toContain("not allowed");
  expect(await currentUser(page)).toMatchObject({ email: state.email });
});

test("an install with an account but no admin gets its owner on the next start", async () => {
  const state = loadState();
  const demoted = await executeSql(`UPDATE platform."UserAuthIdentity" SET role = 'user' WHERE role = 'admin'`);
  expect(demoted).toBe(1);
  const running = app!;
  app = null;
  await quit(running);

  app = await startApp(readyTimeoutMs);
  // The last test compares the installed tree with what it was before this
  // file's first start, so both runs are covered by one check, there.
  app.installed = running.installed;
  const { page, url } = app;
  // A session keeps the role it was created with for a few minutes (the
  // cookie cache in frontend/src/lib/auth/auth.ts), so sign in afresh.
  if (await currentUser(page)) await signOut(page, url);
  await signIn(page, url, state.email, state.password);
  expect(await currentUser(page)).toMatchObject({ email: state.email, role: "admin" });
});

test("quits and leaves nothing behind", async () => {
  const running = app!;
  app = null;
  await quitAndVerify(running);
});
