// Remove the app. The program goes; the user's data stays.

import fs from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

import { verifyFirewall } from "../lib/app";
import { isOwnUninstallName } from "../../src/identity";
import { dataDir, kind, mayUninstall, product } from "../lib/config";
import { appProcesses, platform, run } from "../lib/platform";
import { describe } from "../lib/processes";

test.skip(!mayUninstall, "the app was not installed by these tests (set AUTOGPT_E2E_UNINSTALL=1 to remove it)");

test("uninstalls the app and keeps the data", async () => {
  test.setTimeout(45 * 60_000);
  expect(platform.isInstalled()).toBe(true);
  await platform.uninstall();

  expect(platform.isInstalled()).toBe(false);
  const installDir = platform.installDir();
  if (installDir) expect(fs.existsSync(installDir), `${installDir} is gone`).toBe(false);
  expect(describe(appProcesses()), "processes still running after uninstalling").toBe("");
  // Asked until it is so: the uninstaller takes its entry out of the list
  // last, after the files, and on Windows it is still at work when the
  // program that was started has returned.
  await expect
    .poll(registeredWithSystem, { message: "the system no longer lists the app", timeout: 180_000 })
    .toBe(false);

  // The database, with the account and the agent in it (electron-builder.config.js
  // `deleteAppDataOnUninstall: false`; the other installers never touch it).
  expect(fs.existsSync(path.join(dataDir(), "postgres", "PG_VERSION")), `${dataDir()} is kept`).toBe(true);
  await verifyFirewall();
});

/** Whether the OS still believes the app is installed. */
async function registeredWithSystem(): Promise<boolean> {
  if (kind === "nsis") {
    // Every DisplayName among the uninstall entries; exits with 1 when
    // there is none. The entry is this app's only by its whole name: a
    // search for "AutoGPT" would also find every variant installed here.
    const entries = await run(
      "reg.exe",
      ["query", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall", "/s", "/v", "DisplayName"],
      { check: false },
    );
    const names = [...entries.output.matchAll(/^\s*DisplayName\s+REG_SZ\s+(.*?)\s*$/gm)].map((match) => match[1]);
    return names.some((name) => isOwnUninstallName(product, name));
  }
  if (kind === "deb") {
    const status = await run("dpkg-query", ["-W", "-f=${Status}", product.packageName], { check: false });
    return status.code === 0 && status.output.includes("install ok installed");
  }
  return false;
}
