// Installing, starting and removing the app the way a person does on each
// OS: the silent form of the same installer, the same install location, the
// same executable.

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { UPDATES_OFF, appEnvironment, dataDirOverride, installDirOverride, kind, product } from "./config";
import { isUnder, listProcesses, type RunningProcess } from "./processes";

const INSTALL_TIMEOUT_MS = 40 * 60_000;
const TOOL_TIMEOUT_MS = 5 * 60_000;

/** Facts recorded while installing, shown in the test report. */
export type Notes = Record<string, string>;

export interface Platform {
  /** Install, or upgrade when the app is already installed. */
  install(artifact: string): Promise<Notes>;
  isInstalled(): boolean;
  /** The installed tree, or null where the app is one read-only image. */
  installDir(): string | null;
  executable(): string;
  /** The bundled runtime; the app must be running where it is an image. */
  runtimeDir(): string;
  /** Start the app as its shortcut would, plus `args`, without waiting for
   * it. Whatever the launcher and the shell print goes to `logFile`. */
  launch(args: string[], logFile: string): Promise<void>;
  owns(found: RunningProcess): boolean;
  uninstall(): Promise<void>;
  /** A reason the installed app is no longer what was installed, if any. */
  damage(): Promise<string | null>;
}

export function appProcesses(): RunningProcess[] {
  return listProcesses().filter((found) => platform.owns(found));
}

class WindowsInstall implements Platform {
  // Per-user and one-click (electron-builder.config.js `nsis`): no elevation, no questions.
  private readonly dir =
    installDirOverride ||
    path.join(process.env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "Programs", product.packageName);

  async install(artifact: string): Promise<Notes> {
    // /S installs without starting the app afterwards. /D must come last.
    const args = installDirOverride ? ["/S", `/D=${this.dir}`] : ["/S"];
    // Over an install that is there, from inside it: an update starts the
    // installer from the running app, whose working directory is the install
    // directory when it was started from its shortcut. Windows will not
    // rename a directory a program is standing in, which is how the old
    // version is moved away (resources/installer.nsh). Started from anywhere
    // else, the upgrade measured here would be one no user gets.
    const cwd = fs.existsSync(this.dir) ? this.dir : undefined;
    await run(artifact, args, { timeoutMs: INSTALL_TIMEOUT_MS, cwd });
    const leftover = `${this.dir}.old-install`;
    if (fs.existsSync(leftover)) throw new Error(`the installer left the old version behind in ${leftover}`);
    return cwd ? { "upgrade started from": "the install directory, as an update starts it" } : {};
  }

  isInstalled(): boolean {
    return fs.existsSync(this.executable());
  }

  installDir(): string {
    return this.dir;
  }

  executable(): string {
    return path.join(this.dir, `${product.productFilename}.exe`);
  }

  runtimeDir(): string {
    return path.join(this.dir, "resources", "runtime");
  }

  launch(args: string[], logFile: string): Promise<void> {
    return startDetached(this.executable(), args, logFile);
  }

  owns(found: RunningProcess): boolean {
    return found.executable !== "" && isUnder(found.executable, this.dir);
  }

  async uninstall(): Promise<void> {
    // The uninstaller copies itself to a temporary directory and returns
    // before the copy has finished the work.
    await run(path.join(this.dir, `Uninstall ${product.productFilename}.exe`), ["/S"], { timeoutMs: INSTALL_TIMEOUT_MS });
    // The folder is renamed away before it is deleted (resources/installer.nsh),
    // so its name is free long before its files are gone.
    const moved = `${this.dir}.old-install`;
    await waitUntil(
      () => !fs.existsSync(this.dir) && !fs.existsSync(moved),
      INSTALL_TIMEOUT_MS,
      `${this.dir} to be removed`,
    );
  }

  async damage(): Promise<string | null> {
    return null;
  }
}

class MacInstall implements Platform {
  private readonly app = installDirOverride || `/Applications/${product.productFilename}.app`;

  async install(artifact: string): Promise<Notes> {
    // Mark the image as a browser download would, so that what is recorded
    // below is what a user's Mac would see.
    const stamp = Math.floor(Date.now() / 1000).toString(16);
    await run("xattr", ["-w", "com.apple.quarantine", `0083;${stamp};Safari;${randomUUID()}`, artifact]);
    const mount = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-dmg-"));
    await run("hdiutil", ["attach", "-nobrowse", "-readonly", "-mountpoint", mount, artifact]);
    try {
      // Dragging onto an existing app replaces it; ditto alone would merge.
      fs.rmSync(this.app, { recursive: true, force: true });
      await run("ditto", [path.join(mount, `${product.productFilename}.app`), this.app], { timeoutMs: INSTALL_TIMEOUT_MS });
    } finally {
      await run("hdiutil", ["detach", mount, "-force"], { check: false });
    }
    const signature = await this.damage();
    if (signature) throw new Error(`the installed app fails its signature check:\n${signature}`);
    const gatekeeper = await run("spctl", ["--assess", "--type", "execute", "-vv", this.app], { check: false });
    const quarantine = await run("xattr", ["-p", "com.apple.quarantine", this.app], { check: false });
    // The first-launch dialog cannot be answered without a person. This is
    // what "Open Anyway" does.
    await run("xattr", ["-dr", "com.apple.quarantine", this.app]);
    return {
      "codesign --verify --deep --strict": "passed",
      "spctl --assess": gatekeeper.output.trim(),
      "quarantine attribute after copying": quarantine.output.trim() || "(none)",
    };
  }

  isInstalled(): boolean {
    return fs.existsSync(this.executable());
  }

  installDir(): string {
    return this.app;
  }

  executable(): string {
    return path.join(this.app, "Contents", "MacOS", product.productFilename);
  }

  runtimeDir(): string {
    return path.join(this.app, "Contents", "Resources", "runtime");
  }

  launch(args: string[], logFile: string): Promise<void> {
    const environment = ["--env", UPDATES_OFF];
    if (dataDirOverride) environment.push("--env", `AUTOGPT_DESKTOP_DATA_DIR=${path.resolve(dataDirOverride)}`);
    // `open` returns at once and the app is not its child: it has to be
    // told where the app's own output goes.
    const output = ["--stdout", logFile, "--stderr", logFile];
    return startDetached("open", ["-n", "-a", this.app, ...environment, ...output, "--args", ...args], logFile);
  }

  owns(found: RunningProcess): boolean {
    return isUnder(found.executable, this.app);
  }

  async uninstall(): Promise<void> {
    fs.rmSync(this.app, { recursive: true, force: true });
  }

  /** Anything written into the bundle breaks its signature. */
  async damage(): Promise<string | null> {
    const result = await run("codesign", ["--verify", "--deep", "--strict", "--verbose=2", this.app], {
      check: false,
    });
    return result.code === 0 ? null : result.output;
  }
}

abstract class LinuxInstall {
  abstract executable(): string;

  launch(args: string[], logFile: string): Promise<void> {
    const command = [this.executable(), ...args];
    if (process.env.DISPLAY || process.env.WAYLAND_DISPLAY) {
      return startDetached(command[0], command.slice(1), logFile);
    }
    // No desktop session (CI): a virtual display, and a session bus for the
    // tray icon the app creates at start (src/main.js createTray).
    return startDetached(
      "xvfb-run",
      ["-a", "--server-args=-screen 0 1600x1000x24", "dbus-run-session", "--", ...command],
      logFile,
    );
  }

  async damage(): Promise<string | null> {
    return null;
  }
}

class DebInstall extends LinuxInstall implements Platform {
  // electron-builder installs under the product's name, not the package's.
  private readonly dir = `/opt/${product.productName}`;

  async install(artifact: string): Promise<Notes> {
    await run(
      "sudo",
      ["-n", "env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", path.resolve(artifact)],
      { timeoutMs: INSTALL_TIMEOUT_MS },
    );
    const version = await run("dpkg-query", ["-W", "-f=${Version}", product.packageName]);
    return { "package version": version.output.trim() };
  }

  isInstalled(): boolean {
    return fs.existsSync(this.executable());
  }

  installDir(): string {
    return this.dir;
  }

  executable(): string {
    return path.join(this.dir, product.packageName);
  }

  runtimeDir(): string {
    return path.join(this.dir, "resources", "runtime");
  }

  owns(found: RunningProcess): boolean {
    return isUnder(found.executable, this.dir);
  }

  async uninstall(): Promise<void> {
    await run("sudo", ["-n", "env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "remove", "-y", product.packageName], {
      timeoutMs: INSTALL_TIMEOUT_MS,
    });
  }
}

class AppImageInstall extends LinuxInstall implements Platform {
  private readonly file = path.join(
    installDirOverride || path.join(os.homedir(), "Applications"),
    `${product.artifactBase}.AppImage`,
  );

  async install(artifact: string): Promise<Notes> {
    const fuse = await this.ensureFuse();
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    fs.copyFileSync(artifact, this.file);
    // A download, and a CI artifact, arrive without the executable bit.
    fs.chmodSync(this.file, 0o755);
    return { libfuse2: fuse };
  }

  isInstalled(): boolean {
    return fs.existsSync(this.file);
  }

  installDir(): null {
    return null;
  }

  executable(): string {
    return this.file;
  }

  runtimeDir(): string {
    for (const found of listProcesses()) {
      const mount = ownMount(found.executable);
      if (mount) return path.join(mount, "resources", "runtime");
    }
    throw new Error("the AppImage is not running, so its runtime is not mounted");
  }

  owns(found: RunningProcess): boolean {
    return found.executable === this.file || ownMount(found.executable) !== null;
  }

  async uninstall(): Promise<void> {
    fs.rmSync(this.file, { force: true });
  }

  /** AppImages need FUSE 2, which current Ubuntu no longer installs. */
  private async ensureFuse(): Promise<string> {
    const libraries = await run("ldconfig", ["-p"], { check: false });
    if (libraries.output.includes("libfuse.so.2")) return "already installed";
    const renamed = await run("apt-cache", ["show", "libfuse2t64"], { check: false });
    const name = renamed.code === 0 ? "libfuse2t64" : "libfuse2";
    await run("sudo", ["-n", "env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", name]);
    return `installed ${name}`;
  }
}

// The AppImage runtime mounts the image at <tmp>/.mount_<first six letters
// of the file name><random>; the app's processes run from there. Those six
// letters are the same for the normal app and for every variant, so a mount
// is this app's only if the executable in it is.
const MOUNTED = /^(.*\/\.mount_[^/]+)\//;

function ownMount(executable: string): string | null {
  const mount = MOUNTED.exec(executable)?.[1];
  return mount && fs.existsSync(path.join(mount, product.packageName)) ? mount : null;
}

function create(): Platform {
  if (kind === "nsis") return new WindowsInstall();
  if (kind === "dmg") return new MacInstall();
  if (kind === "deb") return new DebInstall();
  return new AppImageInstall();
}

export const platform: Platform = create();

interface RunOptions {
  timeoutMs?: number;
  /** Throw on a non-zero exit code (the default). */
  check?: boolean;
  /** The working directory; the test runner's when not given. */
  cwd?: string;
}

interface RunResult {
  code: number | null;
  output: string;
}

/** Run a tool to completion without blocking the test runner's event loop. */
export function run(command: string, args: string[], options: RunOptions = {}): Promise<RunResult> {
  const { timeoutMs = TOOL_TIMEOUT_MS, check = true, cwd } = options;
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"], windowsHide: true, cwd });
    let output = "";
    child.stdout.on("data", (chunk) => (output += chunk));
    child.stderr.on("data", (chunk) => (output += chunk));
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${command} did not finish within ${timeoutMs / 1000}s\n${output}`));
    }, timeoutMs);
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (check && code !== 0) {
        reject(new Error(`${command} ${args.join(" ")} exited with ${code}\n${output}`));
      } else {
        resolve({ code, output });
      }
    });
  });
}

export async function waitUntil(condition: () => boolean, timeoutMs: number, what: string): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!condition()) {
    if (Date.now() > deadline) throw new Error(`timed out after ${timeoutMs / 1000}s waiting for ${what}`);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/** Start `command` and let it outlive the tests. Its output is kept: an app
 * that dies before its runtime starts leaves no other trace. */
function startDetached(command: string, args: string[], logFile: string): Promise<void> {
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  const log = fs.openSync(logFile, "a");
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { detached: true, stdio: ["ignore", log, log], env: appEnvironment() });
    child.once("error", (error) => {
      fs.closeSync(log);
      reject(new Error(`could not start ${command}: ${error.message}`));
    });
    child.once("spawn", () => {
      fs.closeSync(log);
      child.unref();
      resolve();
    });
  });
}
