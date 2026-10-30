import { randomBytes } from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { _electron, expect, request, test, type ElectronApplication, type Page } from "@playwright/test";

import { createCalculatorAgent, executionResult, runAndWatch } from "../lib/agent";
import { stopStrayApp } from "../lib/app";
import { appEnvironment, dataDir, firstReadyTimeoutMs, installer, kind, product, readyTimeoutMs } from "../lib/config";
import { appProcesses, platform, run } from "../lib/platform";
import { isAlive, recordedPids } from "../lib/processes";
import { logPosition, logTails, waitForIdle, waitForReady } from "../lib/runtime-log";
import { appVersion, call, completeOnboarding, currentUser, signUp } from "../lib/session";
import { readProofConfig } from "./config";
import { checkFeed, sha256 } from "./releases";

const proof = readProofConfig(process.env);
let ownsInstall = false;
let electron: ElectronApplication | undefined;

test.afterEach(async () => {
  await test.info().attach("runtime-logs", { body: logTails(dataDir()), contentType: "text/plain" });
});

test.afterAll(async () => {
  if (!ownsInstall) return;
  let timer: NodeJS.Timeout | undefined;
  try {
    await Promise.race([
      electron?.close().catch(() => undefined),
      new Promise<void>((resolve) => { timer = setTimeout(resolve, 10_000); }),
    ]);
  } finally {
    clearTimeout(timer);
  }
  await stopStrayApp();
});

test("A discovers and downloads real feed B, then Restart to update preserves its data", async () => {
  expect(installer, "A must be a pinned installer file").not.toBeNull();
  expect(path.resolve(installer!)).toBe(path.resolve(proof.installer));
  expect(fs.statSync(installer!).isFile()).toBe(true);
  expect(platform.isInstalled(), "use a fresh dedicated variant or a disposable normal-app machine").toBe(false);
  const installDirectory = platform.installDir();
  if (installDirectory) expect(fs.existsSync(installDirectory), "the entire installation directory must be unused").toBe(false);
  expect(fs.existsSync(dataDir()), "the default data directory must be unused").toBe(false);
  expect(appProcesses()).toEqual([]);
  expect(await sha256(installer!)).toBe(proof.fromSha256);
  const releases = await checkFeed(proof, path.basename(installer!));
  await platform.install(installer!);
  ownsInstall = true;

  const position = logPosition(dataDir());
  const env = Object.fromEntries(
    Object.entries({ ...appEnvironment(), AUTOGPT_DESKTOP_UPDATES: "on" })
      .filter((entry): entry is [string, string] => typeof entry[1] === "string"),
  );
  if (process.platform === "linux" && !env.DISPLAY && !env.WAYLAND_DISPLAY) {
    throw new Error("Run this opt-in harness under xvfb-run -a dbus-run-session on headless Linux");
  }
  electron = await _electron.launch({ executablePath: platform.executable(), env, timeout: 60_000 });
  const app = electron;
  const url = await waitForReady(dataDir(), position, firstReadyTimeoutMs, {
    isRunning: () => appProcesses().length > 0,
    output: () => "The packaged Electron process is controlled by the opt-in feed harness",
  });
  let page: Page | undefined;
  await expect.poll(() => {
    page = app.windows().find((window) => window.url().startsWith(url));
    return Boolean(page);
  }).toBe(true);
  const window = page!;
  const before = await app.evaluate(({ app }) => {
    const { createRequire } = process.getBuiltinModule("node:module");
    const path = process.getBuiltinModule("node:path");
    const fs = process.getBuiltinModule("node:fs");
    const requireApp = createRequire(path.join(app.getAppPath(), "package.json"));
    const built = requireApp("./package.json");
    const metadata = fs.readFileSync(path.join(process.resourcesPath, "app-update.yml"), "utf8");
    const name = /^updaterCacheDirName: ([A-Za-z0-9_-]+)\r?$/m.exec(metadata)?.[1];
    if (!name) throw new Error("The packaged app does not identify its updater cache");
    const { getAppCacheDir } = requireApp("electron-updater/out/AppAdapter");
    return {
      version: app.getVersion(), packaged: app.isPackaged,
      macDeveloperId: Boolean(built.autogptDesktop?.macDeveloperId),
      pending: path.join(getAppCacheDir(), name, "pending"),
    };
  });
  expect(before.packaged).toBe(true);
  expect(before.version).toBe(proof.fromVersion);
  expect(await appVersion(window)).toBe(proof.fromVersion);
  if (process.platform === "darwin") expect(before.macDeveloperId, "Mac feed installation needs Developer ID A and B from the same team").toBe(true);
  expect(fs.existsSync(before.pending) ? fs.readdirSync(before.pending) : [], "a cached B would not prove a new download").toEqual([]);
  await clickMenu(app, "Check for updates");

  const email = `feed-${randomBytes(5).toString("hex")}@example.com`;
  const password = randomBytes(18).toString("base64url");
  const agentName = `Feed proof ${randomBytes(4).toString("hex")}`;
  await signUp(window, url, email, password);
  expect((await completeOnboarding(window)).status).toBe(200);
  expect(await currentUser(window)).toMatchObject({ email, role: "admin" });
  const ownerId = (await call(window, "GET", "/api/auth/get-session")).body.user.id;
  expect(typeof ownerId).toBe("string");
  const graph = await createCalculatorAgent(window, agentName);
  expect(graph.status).toBe(200);
  const execution = await runAndWatch(window, { id: graph.body.id, version: graph.body.version }, 180_000);
  expect(execution.outcome, JSON.stringify(execution.messages)).toBe("COMPLETED");
  const originalRun = await executionResult(window, graph.body.id, execution.executionId!);
  expect(originalRun.body.status).toBe("COMPLETED");
  const outputs = originalRun.body.node_executions.map((node: { output_data?: unknown }) => node.output_data);
  const sentinel = path.join(dataDir(), "workspaces", "real-feed-proof.txt");
  const workspaceBytes = randomBytes(64);
  fs.writeFileSync(sentinel, workspaceBytes);
  expect(await waitForIdle(dataDir(), position, 30_000)).toBe(0);

  const restartLabel = `Restart to update to ${proof.toVersion}`;
  await expect.poll(() => menuLabels(app), { timeout: 40 * 60_000 }).toContain(restartLabel);
  const candidates = fs.readdirSync(before.pending).filter((name) => /\.(exe|zip|AppImage)$/.test(name));
  const payloads = await Promise.all(candidates.map(async (name) => ({ name, sha256: await sha256(path.join(before.pending, name)) })));
  expect(payloads.some((payload) => payload.sha256 === proof.toSha256), "the actual updater download must match pinned B bytes").toBe(true);
  await checkFeed(proof, path.basename(installer!));

  const oldPids = [...new Set([app.process().pid!, ...appProcesses().map((process) => process.pid), ...recordedPids(dataDir())])];
  const restartPosition = logPosition(dataDir());
  const confirmation = test.info().outputPath("confirmation.json");
  await app.evaluate(({ dialog }, { expected, file }) => {
    dialog.showMessageBox = (async (...args: unknown[]) => {
      const question = args.at(-1) as { message: string; buttons: string[]; detail?: string };
      if (question.message !== expected || question.buttons[0] !== "Restart" || question.detail) {
        throw new Error("Unexpected native dialog or active work during the update proof");
      }
      process.getBuiltinModule("node:fs").writeFileSync(file, JSON.stringify(question));
      return { response: 0, checkboxChecked: false };
    }) as typeof dialog.showMessageBox;
  }, { expected: `Restart ${product.productName} to update to ${proof.toVersion}?`, file: confirmation });
  await clickMenu(app, restartLabel);
  await expect.poll(() => fs.existsSync(confirmation)).toBe(true);
  await expect.poll(() => oldPids.filter(isAlive), { timeout: 120_000 }).toEqual([]);
  const newUrl = await waitForReady(dataDir(), restartPosition, readyTimeoutMs);
  expect(newUrl, "the updater must automatically start B on the original public address").toBe(url);
  expect(appProcesses().length).toBeGreaterThan(0);
  const afterVersion = await installedVersion(proof.toVersion, proof.toSha256);
  expect(afterVersion).toBe(proof.toVersion);
  expect(fs.readFileSync(sentinel).equals(workspaceBytes)).toBe(true);

  const api = await request.newContext({ baseURL: url, extraHTTPHeaders: { Origin: url } });
  try {
    const login = await api.post("/api/auth/sign-in/email", { data: { email, password } });
    expect(login.status(), "the original owner's saved credentials survive").toBe(200);
    const owner = await api.get("/api/auth/get-session");
    expect((await owner.json()).user).toMatchObject({ id: ownerId, email, role: "admin" });
    const savedGraph = await api.get(`/api/proxy/api/graphs/${graph.body.id}`);
    expect(savedGraph.status()).toBe(200);
    expect(await savedGraph.json()).toMatchObject({ id: graph.body.id, name: agentName, nodes: graph.body.nodes });
    const savedRun = await api.get(`/api/proxy/api/graphs/${graph.body.id}/executions/${execution.executionId}`);
    expect(savedRun.status()).toBe(200);
    const result = await savedRun.json();
    expect(result.status).toBe("COMPLETED");
    expect(result.node_executions.map((node: { output_data?: unknown }) => node.output_data)).toEqual(outputs);
  } finally {
    await api.dispose();
  }
  await test.info().attach("A-to-B-proof", {
    body: JSON.stringify({
      releases, from: before.version, to: afterVersion,
      fromSource: proof.fromSource, toSource: proof.toSource,
      fromSha256: proof.fromSha256, toArtifact: proof.toArtifact, payloads,
      publicAddress: url, oldPids, oldPidsGone: true,
      graphId: graph.body.id, executionId: execution.executionId,
      ownerId, ownerPreserved: true, workspacePreserved: true,
    }, null, 2), contentType: "application/json",
  });
});

async function menuLabels(app: ElectronApplication) {
  return app.evaluate(({ Menu }) => {
    function labels(menu: Electron.Menu | null): string[] {
      return menu?.items.flatMap((item) => [item.label, ...labels(item.submenu || null)]) || [];
    }
    return labels(Menu.getApplicationMenu());
  });
}

async function clickMenu(app: ElectronApplication, label: string) {
  await app.evaluate(({ Menu, BrowserWindow }, expected) => {
    function find(menu: Electron.Menu | null): Electron.MenuItem | undefined {
      for (const item of menu?.items || []) {
        if (item.label === expected && item.enabled) return item;
        const nested = find(item.submenu || null);
        if (nested) return nested;
      }
      return undefined;
    }
    const item = find(Menu.getApplicationMenu());
    if (!item) throw new Error(`No enabled menu action: ${expected}`);
    item.click(item, BrowserWindow.getFocusedWindow()!, { triggeredByAccelerator: false });
  }, label);
}

async function installedVersion(expected: string, expectedSha256: string) {
  if (kind === "appimage") {
    expect(await sha256(platform.executable())).toBe(expectedSha256);
    return expected;
  }
  if (process.platform === "darwin") {
    return (await run("/usr/libexec/PlistBuddy", ["-c", "Print :CFBundleShortVersionString", path.join(platform.installDir()!, "Contents", "Info.plist")])).output.trim();
  }
  const literal = platform.executable().replaceAll("'", "''");
  return (await run("powershell.exe", ["-NoProfile", "-Command", `(Get-Item -LiteralPath '${literal}').VersionInfo.ProductVersion`])).output.trim();
}
