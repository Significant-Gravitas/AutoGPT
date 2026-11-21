"use strict";

const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const path = require("node:path");

const MAX_DOCUMENT_BYTES = 64 * 1024;
const TIMEOUT_MS = 1000;
const SERVICES = [
  ["public", "AutoGPT window", "public", "/healthz"],
  ["api", "Workflow API", "agent_api", "/health"],
  ["credentials", "Connection service", "swap_credential", "/health_check"],
  ["frontend", "Interface", "frontend", "/"],
  ["postgres", "Local database", "postgres", null],
  ["valkey", "Local queue cache", "valkey", null],
  ["rabbitmq", "Workflow message broker", "rabbitmq", null],
];
const UNKNOWN = ["unknown", "This check could not be completed."];
const DEFINITIONS = {
  "runtime-state": {
    label: "AutoGPT runtime",
    messages: {
      ready: ["ok", "AutoGPT reports that its runtime is ready."],
      starting: [
        "unknown",
        "AutoGPT is still starting. Refresh Doctor after startup finishes.",
      ],
      stopped: [
        "stopped",
        "AutoGPT is stopped. Local services were not probed.",
      ],
      error: [
        "error",
        "AutoGPT reported a runtime error. Review the startup window before retrying.",
      ],
    },
  },
  "runtime-manifest": {
    label: "Installed runtime",
    messages: {
      compatible: [
        "ok",
        "The runtime manifest and launcher match this platform. This is not a signature or database compatibility check.",
      ],
      missing: [
        "error",
        "The installed runtime manifest is missing. Reinstall AutoGPT from a complete installer.",
      ],
      incompatible: [
        "error",
        "The runtime manifest or launcher does not match this platform. Reinstall AutoGPT from a complete installer.",
      ],
      unreadable: [
        "unknown",
        "The runtime manifest could not be safely read. Check access to the installation.",
      ],
    },
  },
  "disk-space": {
    label: "Space for local data",
    messages: {
      sufficient: [
        "ok",
        "At least 2 GiB of free space is available for local data.",
      ],
      low: [
        "warning",
        "Less than 2 GiB of free space remains. Free space before running large workflows.",
      ],
      critical: [
        "error",
        "Less than 512 MiB of free space remains. Free space before continuing workflows.",
      ],
      missing: [
        "unknown",
        "The local data folder is not available yet. Refresh after AutoGPT starts.",
      ],
    },
  },
  connection: {
    label: "Otto connection",
    messages: {
      connected: [
        "ok",
        "A saved connection is available to Otto. Doctor has not sent a model request.",
      ],
      "not-connected": [
        "warning",
        "No usable saved connection was reported. Open Connections to choose a model connection.",
        "Open Connections",
      ],
      "sign-in-required": [
        "warning",
        "A connection needs sign-in. Open Connections to reconnect it.",
        "Open Connections",
      ],
      unavailable: [
        "warning",
        "The saved connection is currently unavailable. Open Connections to review it.",
        "Open Connections",
      ],
      unknown: [
        "unknown",
        "Saved connection availability has not been verified. Open Connections to review it.",
        "Open Connections",
      ],
    },
  },
  workflow: {
    label: "Workflows",
    messages: {
      available: [
        "available",
        "The runtime and local Workflow API are responding. Individual workflows may need their own connections.",
      ],
      stopped: ["unavailable", "Start AutoGPT to run workflows."],
      unknown: [
        "unknown",
        "Workflow availability has not been confirmed by the local checks.",
      ],
    },
  },
  otto: {
    label: "Otto",
    messages: {
      configured: [
        "configured",
        "A saved model connection is available. A successful Otto response has not been tested.",
      ],
      unavailable: [
        "unavailable",
        "Choose or reconnect a model connection to use Otto.",
        "Open Connections",
      ],
      stopped: ["unavailable", "Start AutoGPT to use Otto."],
      unknown: [
        "unknown",
        "Otto connection readiness has not been verified.",
        "Open Connections",
      ],
    },
  },
  browser: {
    label: "Otto browser tools",
    messages: {
      unavailable: [
        "unavailable",
        "Browser tools are disabled in this desktop build until a managed browser with the required network policy is bundled and verified. Installing a browser on this computer does not enable them.",
      ],
      detected: [
        "unknown",
        "A browser CLI and browser executable were detected on this computer. Otto's effective browser configuration and execution environment have not been verified.",
      ],
      incomplete: [
        "unknown",
        "The browser path visible to Doctor is missing or cannot be used. Otto's effective browser configuration has not been verified. Browser tools are optional.",
      ],
      unknown: [
        "unknown",
        "Browser tools need agent-browser and a browser installation. A complete configuration was not verified; these optional tools are not bundled.",
      ],
    },
  },
  "local-sandbox": {
    label: "Local shell sandbox",
    messages: {
      configured: [
        "configured",
        "Linux bubblewrap was found. Sandbox execution has not been tested.",
      ],
      unavailable: [
        "unavailable",
        "A local shell sandbox is not bundled on Windows or macOS. Shell tools need an enabled cloud sandbox.",
      ],
      missing: [
        "unavailable",
        "Linux bubblewrap was not found. Shell tools need a local sandbox or an enabled cloud sandbox.",
      ],
    },
  },
  "cloud-sandbox": {
    label: "Cloud shell sandbox",
    messages: {
      unknown: [
        "unknown",
        "E2B needs both an enabled configuration and an API key. Doctor does not read secret settings or contact cloud services.",
      ],
    },
  },
  "long-term-memory": {
    label: "Long-term memory",
    messages: {
      unavailable: [
        "unavailable",
        "Graphiti long-term memory is not included in this desktop app. Saved chat history is separate.",
      ],
    },
  },
  bots: {
    label: "Chat bots",
    messages: {
      unavailable: [
        "unavailable",
        "Chat-bot bridge services are not bundled. Adding bot credentials alone does not enable them.",
      ],
    },
  },
  "public-webhooks": {
    label: "Public webhooks",
    messages: {
      unavailable: [
        "unavailable",
        "Public webhook routing is not bundled. The desktop app serves locally on this computer.",
      ],
    },
  },
};
for (const [id, label, , endpoint] of SERVICES) {
  DEFINITIONS[`service-${id}`] = {
    label,
    messages: {
      responding: [
        "ok",
        endpoint
          ? "The local health endpoint responded."
          : "The local port accepts connections. Authentication and stored data were not inspected.",
      ],
      stopped: ["stopped", "AutoGPT is stopped; this service was not probed."],
      starting: ["unknown", "Service checks wait until AutoGPT reports ready."],
      missing: [
        "unknown",
        "No valid saved port is available for this service.",
      ],
      unverified: [
        "unknown",
        "The saved ports could not be matched to the current AutoGPT window. No probe was sent.",
      ],
      failed: [
        "error",
        "The local service did not answer its bounded health check. Refresh Doctor after checking the runtime.",
      ],
      timeout: [
        "warning",
        "The local health check timed out. Refresh Doctor to check again.",
      ],
    },
  };
}
const CHECK_IDS = [
  "runtime-state",
  "runtime-manifest",
  "disk-space",
  ...SERVICES.map(([id]) => `service-${id}`),
  "connection",
];
const CAPABILITY_IDS = [
  "workflow",
  "otto",
  "browser",
  "local-sandbox",
  "cloud-sandbox",
  "long-term-memory",
  "bots",
  "public-webhooks",
];

function own(object, key) {
  if (!object || typeof object !== "object") return undefined;
  return Object.getOwnPropertyDescriptor(object, key)?.value;
}

function member(value, choices, fallback = "unknown") {
  return choices.includes(value) ? value : fallback;
}

function factsFor(id, facts) {
  const output = {};
  const rules =
    id === "disk-space"
      ? { freeMiB: [0, Number.MAX_SAFE_INTEGER] }
      : id.startsWith("service-")
        ? { port: [15000, 31999], httpStatus: [100, 599] }
        : {};
  for (const [key, [minimum, maximum]] of Object.entries(rules)) {
    const value = own(facts, key);
    if (Number.isSafeInteger(value) && value >= minimum && value <= maximum)
      output[key] = value;
  }
  const booleans =
    id === "runtime-manifest"
      ? ["manifestPresent", "launcherPresent"]
      : id === "browser"
        ? ["cliFound", "browserFound"]
        : id === "local-sandbox"
          ? ["executableFound"]
          : [];
  for (const key of booleans)
    if (typeof own(facts, key) === "boolean") output[key] = own(facts, key);
  return output;
}

function item(id, code = "unknown", facts) {
  const definition = DEFINITIONS[id];
  const validCode =
    typeof code === "string" && Object.hasOwn(definition.messages, code)
      ? code
      : "unknown";
  const [status, detail, action] = definition.messages[validCode] || UNKNOWN;
  return {
    id,
    label: definition.label,
    status,
    code: validCode,
    detail,
    ...(action ? { action } : {}),
    facts: factsFor(id, facts),
  };
}

function safeVersion(value) {
  return typeof value === "string" &&
    /^\d{1,6}\.\d{1,6}\.\d{1,6}(?:-(?:dev|alpha|beta|rc|readiness)(?:[.-]\d{1,14})*)?$/.test(
      value,
    )
    ? value
    : "unknown";
}

function scrubDoctorReport(report) {
  const input = own(report, "app");
  const app = {
    version: safeVersion(own(input, "version")),
    platform: member(own(input, "platform"), ["win32", "darwin", "linux"]),
    arch: member(own(input, "arch"), ["x64", "arm64"]),
    runtimeState: member(own(input, "runtimeState"), [
      "starting",
      "ready",
      "stopped",
      "error",
    ]),
  };
  function entries(key, ids) {
    const rows = own(report, key);
    return ids.map((id) => {
      const row = Array.isArray(rows)
        ? rows.find((candidate) => own(candidate, "id") === id)
        : undefined;
      return item(id, own(row, "code"), own(row, "facts"));
    });
  }
  const checks = entries("checks", CHECK_IDS);
  const capabilities = entries("capabilities", CAPABILITY_IDS);
  let summary;
  if (checks.some((check) => check.status === "error"))
    summary = {
      status: "error",
      title: "AutoGPT needs attention",
      detail:
        "Review the reported checks. Doctor has not changed any data or settings.",
    };
  else if (app.runtimeState === "stopped")
    summary = {
      status: "stopped",
      title: "AutoGPT is stopped",
      detail:
        "Local services were not probed. Start AutoGPT and refresh Doctor.",
    };
  else if (app.runtimeState === "starting")
    summary = {
      status: "unknown",
      title: "AutoGPT is starting",
      detail: "Refresh Doctor when startup finishes.",
    };
  else if (
    checks.some(
      (check) => check.status === "warning" || check.status === "unknown",
    )
  )
    summary = {
      status: "warning",
      title: "Some checks need review",
      detail:
        "Unverified and optional capabilities do not mean every feature must be installed.",
    };
  else
    summary = {
      status: "ok",
      title: "Local checks passed",
      detail:
        "Doctor did not run a workflow or contact a model provider. Optional capabilities are listed separately.",
    };
  return { schemaVersion: 1, app, summary, checks, capabilities };
}

function exportDoctorReport(report) {
  return JSON.stringify(scrubDoctorReport(report), null, 2);
}

function inside(root, file) {
  const relative = path.relative(root, file);
  return (
    relative !== ".." &&
    !relative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(relative)
  );
}

function validDirectory(value) {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length < 32768 &&
    !value.includes("\0") &&
    path.isAbsolute(value)
  );
}

async function bounded(operation, milliseconds) {
  let timer;
  try {
    return await Promise.race([
      Promise.resolve().then(operation),
      new Promise((_, reject) => {
        timer = setTimeout(
          () =>
            reject(Object.assign(new Error("timeout"), { code: "ETIMEDOUT" })),
          milliseconds,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function readSmallFile(file) {
  const before = await fs.promises.lstat(file);
  if (!before.isFile() || before.size > MAX_DOCUMENT_BYTES)
    throw new Error("invalid document");
  const handle = await fs.promises.open(
    file,
    fs.constants.O_RDONLY | (fs.constants.O_NONBLOCK || 0),
  );
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size > MAX_DOCUMENT_BYTES)
      throw new Error("invalid document");
    const buffer = Buffer.alloc(MAX_DOCUMENT_BYTES + 1);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    if (bytesRead > MAX_DOCUMENT_BYTES) throw new Error("oversized document");
    return buffer.subarray(0, bytesRead).toString("utf8");
  } finally {
    await handle.close();
  }
}

function probeLoopback({ port, endpoint, timeoutMs }) {
  return new Promise((resolve) => {
    let resource;
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resource?.destroy();
      resolve(result);
    };
    const timer = setTimeout(() => finish({ status: "timeout" }), timeoutMs);
    try {
      if (endpoint) {
        resource = http.get(
          { hostname: "127.0.0.1", port, path: endpoint, agent: false },
          (response) => {
            const httpStatus = response.statusCode;
            response.destroy();
            finish({
              status:
                httpStatus === 200 ||
                (endpoint === "/" && httpStatus >= 300 && httpStatus < 400)
                  ? "responding"
                  : "failed",
              httpStatus,
            });
          },
        );
      } else {
        resource = net.createConnection({ host: "127.0.0.1", port }, () =>
          finish({ status: "responding" }),
        );
      }
      resource.once("error", () => finish({ status: "failed" }));
    } catch {
      finish({ status: "failed" });
    }
  });
}

function createDoctor({
  io = fs.promises,
  readText = readSmallFile,
  probe = probeLoopback,
  env = process.env,
  timeoutMs = TIMEOUT_MS,
} = {}) {
  const limit = Number.isFinite(timeoutMs)
    ? Math.max(1, Math.min(timeoutMs, TIMEOUT_MS))
    : TIMEOUT_MS;
  const call = (operation) => bounded(operation, limit);
  async function document(directory, relative) {
    if (!validDirectory(directory)) return { status: "unknown" };
    try {
      const root = await call(() => io.realpath(directory));
      const file = await call(() => io.realpath(path.join(root, relative)));
      if (!inside(root, file)) return { status: "unreadable" };
      const text = await call(() => readText(file));
      if (
        typeof text !== "string" ||
        Buffer.byteLength(text) > MAX_DOCUMENT_BYTES
      )
        return { status: "unreadable" };
      const value = JSON.parse(text);
      return value && typeof value === "object" && !Array.isArray(value)
        ? { status: "ok", value }
        : { status: "unreadable" };
    } catch (error) {
      return { status: error?.code === "ENOENT" ? "missing" : "unreadable" };
    }
  }
  async function executable(file, platform) {
    if (!validDirectory(file)) return false;
    try {
      const stat = await call(() => io.stat(file));
      if (!stat.isFile()) return false;
      await call(() =>
        io.access(
          file,
          platform === "win32" ? fs.constants.F_OK : fs.constants.X_OK,
        ),
      );
      return true;
    } catch (error) {
      return error?.code === "ENOENT" ? false : null;
    }
  }
  async function findExecutable(name, runtimeDir, platform) {
    const directories = [
      validDirectory(runtimeDir) ? path.join(runtimeDir, "tools", "bin") : null,
      ...(typeof env.PATH === "string"
        ? env.PATH.split(path.delimiter).slice(0, 64)
        : []),
    ].filter(validDirectory);
    const names =
      platform === "win32"
        ? [`${name}.exe`, `${name}.cmd`, `${name}.bat`, name]
        : [name];
    const results = await Promise.all(
      directories.flatMap((directory) =>
        names.map((file) => executable(path.join(directory, file), platform)),
      ),
    );
    return results.includes(true)
      ? true
      : results.includes(null)
        ? null
        : false;
  }
  async function manifestCheck(runtimeDir, platform) {
    const result = await document(runtimeDir, "manifest.json");
    if (result.status !== "ok") return item("runtime-manifest", result.status);
    if (!["win32", "darwin", "linux"].includes(platform))
      return item("runtime-manifest", "unknown", { manifestPresent: true });
    const entry = own(result.value, platform) || own(result.value, "default");
    const command = own(entry, "command");
    const expected =
      platform === "win32" ? "python/python.exe" : "python/bin/python3";
    const args = own(entry, "args");
    const expectedArgs = ["-B", "-m", "autogpt_desktop", "serve"];
    if (
      command !== expected ||
      !Array.isArray(args) ||
      args.length !== expectedArgs.length ||
      !args.every((arg, index) => arg === expectedArgs[index])
    ) {
      return item("runtime-manifest", "incompatible", {
        manifestPresent: true,
      });
    }
    try {
      const root = await call(() => io.realpath(runtimeDir));
      const target = await call(() => io.realpath(path.join(root, command)));
      const present =
        inside(root, target) && (await executable(target, platform));
      return item(
        "runtime-manifest",
        present === null
          ? "unreadable"
          : present
            ? "compatible"
            : "incompatible",
        { manifestPresent: true, launcherPresent: present },
      );
    } catch (error) {
      return item(
        "runtime-manifest",
        error?.code === "ENOENT" ? "incompatible" : "unreadable",
        { manifestPresent: true },
      );
    }
  }
  async function diskCheck(dataDir) {
    if (!validDirectory(dataDir)) return item("disk-space");
    try {
      const disk = await call(() => io.statfs(dataDir, { bigint: true }));
      if (
        !["number", "bigint"].includes(typeof disk.bavail) ||
        !["number", "bigint"].includes(typeof disk.bsize)
      )
        return item("disk-space");
      const available = Number(disk.bavail) * Number(disk.bsize);
      if (!Number.isFinite(available) || available < 0)
        return item("disk-space");
      const freeMiB = Math.floor(available / (1024 * 1024));
      return item(
        "disk-space",
        freeMiB < 512 ? "critical" : freeMiB < 2048 ? "low" : "sufficient",
        { freeMiB },
      );
    } catch (error) {
      return item(
        "disk-space",
        error?.code === "ENOENT" ? "missing" : "unknown",
      );
    }
  }
  async function serviceChecks(dataDir, appUrl, runtimeState) {
    if (runtimeState !== "ready")
      return SERVICES.map(([id]) =>
        item(
          `service-${id}`,
          runtimeState === "stopped" ? "stopped" : "starting",
        ),
      );
    const ports = await document(dataDir, path.join("config", "ports.json"));
    const validPort = (value) =>
      Number.isInteger(value) && value >= 15000 && value < 32000;
    let matched = false;
    try {
      const url = new URL(typeof appUrl === "string" ? appUrl : "");
      matched =
        ports.status === "ok" &&
        url.protocol === "http:" &&
        url.hostname === "127.0.0.1" &&
        !url.username &&
        !url.password &&
        !url.search &&
        !url.hash &&
        url.pathname === "/" &&
        validPort(own(ports.value, "public")) &&
        Number(url.port) === own(ports.value, "public");
    } catch {}
    return Promise.all(
      SERVICES.map(async ([id, , key, endpoint]) => {
        if (ports.status === "missing") return item(`service-${id}`, "missing");
        if (!matched) return item(`service-${id}`, "unverified");
        const port = own(ports.value, key);
        if (
          !validPort(port) ||
          SERVICES.filter(
            ([, , candidate]) => own(ports.value, candidate) === port,
          ).length !== 1
        )
          return item(`service-${id}`, "missing");
        try {
          const result = await call(() =>
            probe({ host: "127.0.0.1", port, endpoint, timeoutMs: limit }),
          );
          return item(
            `service-${id}`,
            member(own(result, "status"), ["responding", "failed", "timeout"]),
            { port, httpStatus: own(result, "httpStatus") },
          );
        } catch (error) {
          return item(
            `service-${id}`,
            error?.code === "ETIMEDOUT" ? "timeout" : "failed",
            { port },
          );
        }
      }),
    );
  }
  return async function collect(options = {}) {
    const dataDir = own(options, "dataDir");
    const runtimeDir = own(options, "runtimeDir");
    const platform = member(own(options, "platform"), [
      "win32",
      "darwin",
      "linux",
    ]);
    const runtimeState = member(own(options, "runtimeState"), [
      "starting",
      "ready",
      "stopped",
      "error",
    ]);
    const suppliedConnection = own(options, "connectionStatus");
    const connection = member(
      typeof suppliedConnection === "string"
        ? suppliedConnection
        : own(suppliedConnection, "status"),
      ["connected", "not-connected", "sign-in-required", "unavailable"],
    );
    const [manifest, disk, services, cliFound, browserFound, sandboxFound] =
      await Promise.all([
        manifestCheck(runtimeDir, platform),
        diskCheck(dataDir),
        serviceChecks(dataDir, own(options, "appUrl"), runtimeState),
        findExecutable("agent-browser", runtimeDir, platform),
        executable(env.AGENT_BROWSER_EXECUTABLE_PATH, platform),
        platform === "linux"
          ? findExecutable("bwrap", runtimeDir, platform)
          : false,
      ]);
    const capabilities = [
      item(
        "workflow",
        runtimeState === "stopped"
          ? "stopped"
          : runtimeState === "ready" &&
              services.find((check) => check.id === "service-api")?.status ===
                "ok"
            ? "available"
            : "unknown",
      ),
      item(
        "otto",
        runtimeState === "stopped"
          ? "stopped"
          : connection === "connected"
            ? "configured"
            : connection === "unknown"
              ? "unknown"
              : "unavailable",
      ),
      item(
        "browser",
        "unavailable",
        { cliFound, browserFound },
      ),
      item(
        "local-sandbox",
        platform === "linux"
          ? sandboxFound === null
            ? "unknown"
            : sandboxFound
              ? "configured"
              : "missing"
          : platform === "unknown"
            ? "unknown"
            : "unavailable",
        { executableFound: sandboxFound },
      ),
      item("cloud-sandbox"),
      item("long-term-memory", "unavailable"),
      item("bots", "unavailable"),
      item("public-webhooks", "unavailable"),
    ];
    return scrubDoctorReport({
      app: {
        version: own(options, "version"),
        platform,
        arch: own(options, "arch"),
        runtimeState,
      },
      checks: [
        item("runtime-state", runtimeState),
        manifest,
        disk,
        ...services,
        item("connection", connection),
      ],
      capabilities,
    });
  };
}

async function collectDoctorReport(options) {
  return createDoctor()(options);
}

module.exports = {
  collectDoctorReport,
  createDoctor,
  scrubDoctorReport,
  exportDoctorReport,
};
