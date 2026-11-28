// Everything the tests take from the environment. README.md documents each
// variable; nothing else in the harness reads process.env for configuration.

import fs from "node:fs";
import path from "node:path";

import { identity } from "../../src/identity";
import { defaultDataDir } from "../../src/paths";

/** Which app is under test: the normal one, or the variant the installers
 * were built as. Every name and path below comes from the definition the
 * build itself uses (src/identity.js). */
export const product = identity(process.env.AUTOGPT_DESKTOP_VARIANT || "");

export type Kind = "nsis" | "dmg" | "deb" | "appimage";

const EXTENSIONS: Record<Kind, string> = {
  nsis: ".exe",
  dmg: ".dmg",
  deb: ".deb",
  appimage: ".AppImage",
};

export const kind: Kind = readKind();

/** The installer under test, or null to test the app that is already installed. */
export const installer = resolveInstaller(process.env.AUTOGPT_E2E_INSTALLER);

/** A build with a higher version, for the upgrade test. */
export const upgradeInstaller = resolveInstaller(process.env.AUTOGPT_E2E_UPGRADE_INSTALLER);

/** Where the app is (to be) installed, when it is not the installer's default. */
export const installDirOverride = process.env.AUTOGPT_E2E_INSTALL_DIR || null;

/** The uninstall test removes the app only if these tests installed it, or on request. */
export const mayUninstall = installer !== null || process.env.AUTOGPT_E2E_UNINSTALL === "1";

/** Changes this machine's firewall and audit settings; for throwaway machines. */
export const checkFirewall = process.platform === "win32" && process.env.AUTOGPT_E2E_FIREWALL === "1";

export const firstReadyTimeoutMs = seconds("AUTOGPT_E2E_FIRST_READY_SECONDS", 600);
export const readyTimeoutMs = seconds("AUTOGPT_E2E_READY_SECONDS", 300);

/** Set when the app must not use its default data directory. */
export const dataDirOverride =
  process.env.AUTOGPT_E2E_DATA_DIR || process.env.AUTOGPT_DESKTOP_DATA_DIR || null;

/** Where the app under test keeps its data, by the app's own rule
 * (src/paths.js). A variant that is given a directory uses the one next to
 * it, `<directory>-<slug>`: the directory itself is the normal app's. */
export function dataDir(): string {
  return path.resolve(defaultDataDir(process.platform, appEnvironment(), product));
}

/** The app under test must not look for updates: a release build would
 * fetch the latest release from GitHub in the middle of the tests. */
export const UPDATES_OFF = "AUTOGPT_DESKTOP_UPDATES=off";

/** What the app is started with: the caller's environment, minus what would
 * point a packaged app at a development runtime. */
export function appEnvironment(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  env.AUTOGPT_DESKTOP_UPDATES = "off";
  delete env.AUTOGPT_DESKTOP_RUNTIME;
  delete env.AUTOGPT_DESKTOP_DATA_DIR;
  delete env.ELECTRON_RUN_AS_NODE;
  if (dataDirOverride) env.AUTOGPT_DESKTOP_DATA_DIR = path.resolve(dataDirOverride);
  return env;
}

function readKind(): Kind {
  const configured = (process.env.AUTOGPT_E2E_KIND || "").toLowerCase();
  if (configured) {
    if (!(configured in EXTENSIONS)) {
      throw new Error(`AUTOGPT_E2E_KIND must be one of ${Object.keys(EXTENSIONS).join(", ")}`);
    }
    return configured as Kind;
  }
  if (process.platform === "win32") return "nsis";
  if (process.platform === "darwin") return "dmg";
  return "deb";
}

/** A file, or a directory holding exactly one installer of this kind. */
function resolveInstaller(value: string | undefined): string | null {
  if (!value) return null;
  const target = path.resolve(value);
  if (!fs.existsSync(target)) throw new Error(`no installer at ${target}`);
  if (!fs.statSync(target).isDirectory()) return target;
  const extension = EXTENSIONS[kind];
  const found = fs.readdirSync(target).filter((name) => name.endsWith(extension));
  if (found.length !== 1) {
    throw new Error(`expected one ${extension} in ${target}, found ${found.length}: ${found.join(", ")}`);
  }
  return path.join(target, found[0]);
}

function seconds(name: string, fallback: number): number {
  const value = Number(process.env[name] || fallback);
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a number of seconds`);
  return value * 1000;
}
