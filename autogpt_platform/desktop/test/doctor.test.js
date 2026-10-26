"use strict";

const assert = require("node:assert/strict");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");
const {
  createDoctor,
  scrubDoctorReport,
  exportDoctorReport,
} = require("../src/doctor");

function fixture(t, platform = "win32") {
  const root = fs.mkdtempSync(
    path.join(os.tmpdir(), "autogpt-doctor-fixture-"),
  );
  t.after(() => {
    assert.ok(
      path.dirname(root) === os.tmpdir() &&
        path.basename(root).startsWith("autogpt-doctor-fixture-"),
    );
    fs.rmSync(root, { recursive: true, force: true });
  });
  const dataDir = path.join(root, "private-owner-data");
  const runtimeDir = path.join(root, "runtime");
  fs.mkdirSync(path.join(dataDir, "config"), { recursive: true });
  const command =
    platform === "win32" ? "python/python.exe" : "python/bin/python3";
  const launcher = path.join(runtimeDir, command);
  fs.mkdirSync(path.dirname(launcher), { recursive: true });
  fs.writeFileSync(launcher, "synthetic launcher; never executed", {
    mode: 0o755,
  });
  fs.writeFileSync(
    path.join(runtimeDir, "manifest.json"),
    JSON.stringify({
      [platform]: { command, args: ["-B", "-m", "autogpt_desktop", "serve"] },
    }),
  );
  const ports = {
    public: 18473,
    agent_api: 18474,
    frontend: 18475,
    postgres: 18476,
    valkey: 18477,
    rabbitmq: 18478,
    swap_credential: 18479,
  };
  const portsFile = path.join(dataDir, "config", "ports.json");
  fs.writeFileSync(portsFile, JSON.stringify(ports));
  const calls = [];
  const options = {
    dataDir,
    runtimeDir,
    version: "0.1.6-readiness.20261020",
    platform,
    arch: "x64",
    appUrl: "http://127.0.0.1:18473",
    runtimeState: "ready",
    connectionStatus: "connected",
  };
  const dependencies = {
    env: {},
    io: {
      ...fs.promises,
      statfs: async () => ({ bavail: 4096n, bsize: 1024n * 1024n }),
    },
    probe: async (request) => {
      calls.push(request);
      return {
        status: "responding",
        httpStatus: request.endpoint ? 200 : undefined,
      };
    },
  };
  return {
    root,
    dataDir,
    runtimeDir,
    ports,
    portsFile,
    calls,
    options,
    dependencies,
    collect: (overrides = {}, deps = {}) =>
      createDoctor({ ...dependencies, ...deps })({ ...options, ...overrides }),
  };
}

function check(report, id) {
  return [...report.checks, ...report.capabilities].find(
    (entry) => entry.id === id,
  );
}

test("ready reports local checks without treating optional capabilities as failures", async (t) => {
  const f = fixture(t);
  const report = await f.collect();
  assert.equal(report.summary.status, "ok");
  assert.equal(report.app.version, f.options.version);
  assert.equal(check(report, "runtime-manifest").code, "compatible");
  assert.equal(check(report, "workflow").status, "available");
  assert.equal(check(report, "otto").status, "configured");
  assert.match(check(report, "otto").detail, /has not been tested/);
  for (const id of [
    "long-term-memory",
    "bots",
    "public-webhooks",
    "local-sandbox",
  ])
    assert.equal(check(report, id).status, "unavailable");
  assert.equal(check(report, "cloud-sandbox").status, "unknown");
  assert.equal(f.calls.length, 7);
  assert.equal(check(report, "service-credentials").status, "ok");
  assert.ok(
    f.calls.every(
      ({ host, timeoutMs }) => host === "127.0.0.1" && timeoutMs <= 1000,
    ),
  );
  assert.deepEqual(
    f.calls.filter(({ endpoint }) => endpoint).map(({ endpoint }) => endpoint),
    ["/healthz", "/health", "/health_check", "/"],
  );
  assert.match(
    check(report, "service-postgres").detail,
    /Authentication and stored data were not inspected/,
  );
});

for (const runtimeState of ["stopped", "starting", "error", "garbage"]) {
  test(`${runtimeState}: do not probe saved ports when runtime is not ready`, async (t) => {
    const f = fixture(t);
    const report = await f.collect({ runtimeState });
    assert.deepEqual(f.calls, []);
    assert.equal(
      check(report, "service-public").status,
      runtimeState === "stopped" ? "stopped" : "unknown",
    );
    if (runtimeState === "stopped")
      assert.equal(report.summary.status, "stopped");
  });
}

for (const appUrl of [
  "https://example.com",
  "http://127.0.0.1:19999",
  "http://127.0.0.1:18473/?secret=do-not-export",
  "http://user:secret@127.0.0.1:18473",
  "http://[::1]:18473",
  null,
]) {
  test(`an unverified current origin prevents all service probes: ${String(appUrl)}`, async (t) => {
    const f = fixture(t);
    const report = await f.collect({ appUrl });
    assert.deepEqual(f.calls, []);
    assert.equal(check(report, "service-public").code, "unverified");
    assert.doesNotMatch(
      exportDoctorReport(report),
      /do-not-export|user:secret|example\.com/,
    );
  });
}

test("missing, malformed, duplicate and out-of-range saved ports are distinguished", async (t) => {
  const f = fixture(t);
  fs.unlinkSync(f.portsFile);
  assert.equal(check(await f.collect(), "service-public").code, "missing");
  fs.writeFileSync(f.portsFile, "private-broken-json");
  assert.equal(check(await f.collect(), "service-public").code, "unverified");
  fs.writeFileSync(
    f.portsFile,
    JSON.stringify({ ...f.ports, postgres: 22, valkey: f.ports.rabbitmq }),
  );
  const report = await f.collect();
  for (const id of ["postgres", "valkey", "rabbitmq"])
    assert.equal(check(report, `service-${id}`).code, "missing");
  assert.equal(f.calls.length, 4);
});

test("manifest checks never execute a launcher or follow one outside the bundle", async (t) => {
  const f = fixture(t);
  const manifest = path.join(f.runtimeDir, "manifest.json");
  for (const command of [
    "../private-key",
    "C:\\private\\launcher.exe",
    "https://example.com/tool",
    "python/bin/python3",
  ]) {
    fs.writeFileSync(
      manifest,
      JSON.stringify({ win32: { command, args: [] } }),
    );
    assert.equal(
      check(await f.collect(), "runtime-manifest").code,
      "incompatible",
    );
  }
  fs.writeFileSync(
    manifest,
    JSON.stringify({
      win32: { command: "python/python.exe", args: ["-c", "SECRET_COMMAND"] },
    }),
  );
  assert.equal(
    check(await f.collect(), "runtime-manifest").code,
    "incompatible",
  );
  fs.writeFileSync(manifest, "x".repeat(64 * 1024 + 1));
  assert.equal(check(await f.collect(), "runtime-manifest").code, "unreadable");
  fs.unlinkSync(manifest);
  assert.equal(check(await f.collect(), "runtime-manifest").code, "missing");
});

test("linked metadata outside its designated directory is not read", async (t) => {
  const f = fixture(t);
  const runtimeDir = path.join(f.root, "runtime-alias");
  fs.symlinkSync(
    f.runtimeDir,
    runtimeDir,
    process.platform === "win32" ? "junction" : "dir",
  );
  const manifest = path.join(
    await fs.promises.realpath(runtimeDir),
    "manifest.json",
  );
  const outside = path.join(
    await fs.promises.realpath(f.root),
    "outside-manifest.json",
  );
  fs.copyFileSync(manifest, outside);
  const reads = [];
  let redirects = 0;
  const realpath = async (file) => {
    if (file === manifest) {
      redirects++;
      return outside;
    }
    return fs.promises.realpath(file);
  };
  const report = await f.collect(
    { runtimeDir },
    {
      io: { ...f.dependencies.io, realpath },
      readText: async (file) => {
        reads.push(file);
        return fs.promises.readFile(file, "utf8");
      },
    },
  );
  assert.equal(check(report, "runtime-manifest").code, "unreadable");
  assert.equal(redirects, 1);
  assert.deepEqual(reads, [await fs.promises.realpath(f.portsFile)]);
});

for (const outside of [false, true]) {
  test(`profile metadata directory links ${outside ? "outside the profile are rejected" : "inside the profile are readable"}`, async (t) => {
    const f = fixture(t);
    const dataDir = path.join(f.root, "profile-alias");
    const target = outside
      ? `${f.dataDir}-outside`
      : path.join(f.dataDir, "saved-config");
    const linkType = process.platform === "win32" ? "junction" : "dir";
    fs.mkdirSync(target);
    fs.writeFileSync(path.join(target, "ports.json"), JSON.stringify(f.ports));
    fs.unlinkSync(f.portsFile);
    fs.rmdirSync(path.dirname(f.portsFile));
    fs.symlinkSync(target, path.dirname(f.portsFile), linkType);
    fs.symlinkSync(f.dataDir, dataDir, linkType);
    const ports = await fs.promises.realpath(
      path.join(dataDir, "config", "ports.json"),
    );
    assert.equal(
      ports,
      await fs.promises.realpath(path.join(target, "ports.json")),
    );
    const reads = [];
    const report = await f.collect(
      { dataDir },
      {
        readText: async (file) => {
          reads.push(file);
          return fs.promises.readFile(file, "utf8");
        },
      },
    );
    assert.equal(check(report, "runtime-manifest").code, "compatible");
    assert.equal(
      check(report, "service-public").code,
      outside ? "unverified" : "responding",
    );
    assert.deepEqual(
      reads.sort(),
      [
        await fs.promises.realpath(path.join(f.runtimeDir, "manifest.json")),
        ...(outside ? [] : [ports]),
      ].sort(),
    );
    assert.equal(f.calls.length, outside ? 0 : 7);
  });
}

test("disk space is read-only and handles low, unknown and unavailable filesystems", async (t) => {
  const f = fixture(t);
  for (const [free, code] of [
    [4096, "sufficient"],
    [1024, "low"],
    [100, "critical"],
  ]) {
    const report = await f.collect(
      {},
      {
        io: {
          ...f.dependencies.io,
          statfs: async () => ({ bavail: BigInt(free), bsize: 1048576n }),
        },
      },
    );
    assert.equal(check(report, "disk-space").code, code);
    assert.equal(check(report, "disk-space").facts.freeMiB, free);
  }
  for (const [code, expected] of [
    ["ENOENT", "missing"],
    ["EACCES", "unknown"],
    ["ENOSYS", "unknown"],
  ]) {
    const report = await f.collect(
      {},
      {
        io: {
          ...f.dependencies.io,
          statfs: async () => {
            throw Object.assign(new Error("SECRET_PATH"), { code });
          },
        },
      },
    );
    assert.equal(check(report, "disk-space").code, expected);
    assert.doesNotMatch(exportDoctorReport(report), /SECRET_PATH/);
  }
});

test("connection availability is allowlisted and never claims successful inference", async (t) => {
  const f = fixture(t);
  const report = await f.collect({
    connectionStatus: {
      status: "sign-in-required",
      token: "PRIVATE_TOKEN",
      email: "private@example.com",
    },
  });
  assert.equal(check(report, "connection").status, "warning");
  assert.equal(check(report, "connection").action, "Open Connections");
  assert.equal(check(report, "otto").status, "unavailable");
  assert.doesNotMatch(
    exportDoctorReport(report),
    /PRIVATE_TOKEN|private@example/,
  );
  assert.equal(
    check(await f.collect({ connectionStatus: "SECRET_ERROR" }), "connection")
      .status,
    "unknown",
  );
});

test("desktop browser tools stay unavailable even with installed executables", async (t) => {
  const f = fixture(t, "linux");
  const bin = path.join(f.runtimeDir, "tools", "bin");
  fs.mkdirSync(bin, { recursive: true });
  for (const name of ["agent-browser", "chromium", "bwrap"])
    fs.writeFileSync(path.join(bin, name), "fixture", { mode: 0o755 });
  const env = { AGENT_BROWSER_EXECUTABLE_PATH: path.join(bin, "chromium") };
  const report = await f.collect({}, { env });
  assert.equal(check(report, "browser").status, "unavailable");
  assert.equal(check(report, "browser").code, "unavailable");
  assert.match(
    check(report, "browser").detail,
    /disabled.*managed browser/,
  );
  assert.deepEqual(check(report, "browser").facts, {
    cliFound: true,
    browserFound: true,
  });
  assert.equal(check(report, "local-sandbox").status, "configured");
  assert.match(check(report, "local-sandbox").detail, /has not been tested/);
  fs.unlinkSync(path.join(bin, "chromium"));
  assert.equal(
    check(await f.collect({}, { env }), "browser").status,
    "unavailable",
  );
  assert.equal(
    check(await f.collect({}, { env: {} }), "browser").status,
    "unavailable",
  );
});

test("host PATH cannot enable desktop browser tools or expose secret settings", async (t) => {
  const f = fixture(t, "linux");
  const hostBin = path.join(f.root, "host-tools");
  fs.mkdirSync(hostBin);
  for (const name of ["agent-browser", "chromium"])
    fs.writeFileSync(path.join(hostBin, name), "fixture", { mode: 0o755 });
  const report = await f.collect(
    {},
    {
      env: {
        PATH: hostBin,
        AGENT_BROWSER_EXECUTABLE_PATH: path.join(hostBin, "chromium"),
      },
      readText: async (file) => {
        assert.ok(
          [path.join(f.runtimeDir, "manifest.json"), f.portsFile].includes(
            file,
          ),
        );
        return fs.promises.readFile(file, "utf8");
      },
    },
  );
  assert.equal(check(report, "browser").status, "unavailable");
  assert.equal(check(report, "browser").code, "unavailable");
  assert.deepEqual(check(report, "browser").facts, {
    cliFound: true,
    browserFound: true,
  });
  assert.doesNotMatch(
    exportDoctorReport(report),
    /host-tools|private-owner-data/,
  );
});

test("secret settings, logs, databases and unrelated environment entries are never read", async (t) => {
  const f = fixture(t);
  const allowed = new Set([
    path.join(f.runtimeDir, "manifest.json"),
    f.portsFile,
  ]);
  const env = new Proxy(
    {},
    {
      get(_, key) {
        assert.ok(["PATH", "AGENT_BROWSER_EXECUTABLE_PATH"].includes(key));
        return undefined;
      },
    },
  );
  const report = await f.collect(
    {},
    {
      env,
      readText: async (file) => {
        assert.ok(allowed.has(file));
        return fs.promises.readFile(file, "utf8");
      },
    },
  );
  assert.equal(check(report, "cloud-sandbox").status, "unknown");
});

test("hanging I/O and probes return bounded unknown/timeout results without error details", async (t) => {
  const f = fixture(t);
  const started = Date.now();
  const report = await f.collect(
    {},
    {
      timeoutMs: 15,
      io: { ...f.dependencies.io, statfs: () => new Promise(() => {}) },
      probe: () => new Promise(() => {}),
    },
  );
  assert.ok(Date.now() - started < 1500);
  assert.equal(check(report, "disk-space").status, "unknown");
  assert.equal(check(report, "service-public").code, "timeout");
});

test("export reconstructs a scrubbed allowlist instead of trusting preview strings", async (t) => {
  const f = fixture(t);
  const report = await f.collect({
    version: "C:\\Users\\PrivateOwner\\token.txt",
    platform: "private-host",
    arch: "private-arch",
  });
  report.password = "TOP_SECRET";
  report.summary.detail = "TOP_SECRET";
  report.app.username = "PrivateOwner";
  for (const row of [...report.checks, ...report.capabilities]) {
    row.label = "TOP_SECRET";
    row.detail = "TOP_SECRET";
    row.action = "TOP_SECRET";
    row.facts.password = "TOP_SECRET";
    row.facts.freeMiB = "TOP_SECRET";
  }
  report.checks.push({ id: "custom-private", code: "TOP_SECRET" });
  const json = exportDoctorReport(report);
  assert.doesNotMatch(
    json,
    /TOP_SECRET|PrivateOwner|private-host|private-arch|token\.txt|custom-private|password|username/,
  );
  assert.doesNotMatch(
    json,
    new RegExp(f.root.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
  );
  assert.equal(JSON.parse(json).app.version, "unknown");
  assert.deepEqual(JSON.parse(json), scrubDoctorReport(JSON.parse(json)));
  assert.doesNotThrow(() =>
    exportDoctorReport({
      app: {
        get version() {
          throw new Error("do not read");
        },
      },
      checks: [
        {
          id: "connection",
          code: {
            toString() {
              throw new Error("do not call");
            },
          },
        },
      ],
    }),
  );
});

function networkDoctor(httpStub, netStub, fsStub = fs) {
  const module = { exports: {} };
  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "../src/doctor.js"), "utf8"),
    {
      require(name) {
        return name === "node:http"
          ? httpStub
          : name === "node:net"
            ? netStub
            : name === "node:fs"
              ? fsStub
              : require(name);
      },
      module,
      exports: module.exports,
      Buffer,
      URL,
      process: { env: {} },
      setTimeout,
      clearTimeout,
    },
  );
  return module.exports;
}

test("metadata rejects non-regular files before opening them", async (t) => {
  const f = fixture(t);
  for (const stat of [
    { isFile: () => false, size: 0 },
    { isFile: () => true, size: 65537 },
  ]) {
    const doctor = networkDoctor(
      {},
      {},
      {
        ...fs,
        promises: {
          ...fs.promises,
          lstat: async () => stat,
          open: async () => assert.fail("special or oversized file was opened"),
        },
      },
    );
    const report = await doctor.createDoctor(f.dependencies)({
      ...f.options,
      runtimeState: "stopped",
    });
    assert.equal(check(report, "runtime-manifest").code, "unreadable");
  }
});

test("metadata opens nonblocking and rechecks the opened handle before reading", async (t) => {
  const f = fixture(t);
  let opened = 0;
  let closed = 0;
  const doctor = networkDoctor(
    {},
    {},
    {
      ...fs,
      constants: { ...fs.constants, O_NONBLOCK: 2048 },
      promises: {
        ...fs.promises,
        lstat: async () => ({ isFile: () => true, size: 0 }),
        open: async (_file, flags) => {
          opened += 1;
          assert.equal(flags, fs.constants.O_RDONLY | 2048);
          return {
            stat: async () => ({ isFile: () => false, size: 0 }),
            read: async () => assert.fail("changed non-regular file was read"),
            close: async () => {
              closed += 1;
            },
          };
        },
      },
    },
  );
  const report = await doctor.createDoctor(f.dependencies)({
    ...f.options,
    runtimeState: "stopped",
  });
  assert.equal(check(report, "runtime-manifest").code, "unreadable");
  assert.equal(opened, 1);
  assert.equal(closed, 1);
});

test("real probe implementation uses fixed loopback, never follows redirects, and closes sockets", async (t) => {
  const f = fixture(t);
  const resources = [];
  const requests = [];
  function resource() {
    const found = new EventEmitter();
    found.destroyed = false;
    found.destroy = () => {
      found.destroyed = true;
    };
    resources.push(found);
    return found;
  }
  const doctor = networkDoctor(
    {
      get(options, callback) {
        requests.push(options);
        const request = resource();
        queueMicrotask(() =>
          callback({
            statusCode: 302,
            headers: { location: "https://private.example/token" },
            destroy() {},
          }),
        );
        return request;
      },
    },
    {
      createConnection(options, callback) {
        requests.push(options);
        const socket = resource();
        queueMicrotask(callback);
        return socket;
      },
    },
  );
  const { probe, ...dependencies } = f.dependencies;
  const report = await doctor.createDoctor(dependencies)(f.options);
  assert.equal(requests.length, 7);
  assert.ok(
    requests.every(
      (request) => (request.host || request.hostname) === "127.0.0.1",
    ),
  );
  assert.ok(resources.every(({ destroyed }) => destroyed));
  assert.equal(check(report, "service-public").code, "failed");
  assert.equal(check(report, "service-frontend").code, "responding");
  assert.doesNotMatch(JSON.stringify(report), /private\.example|token/);
});

test("real probe timeout destroys its owned resources without touching any process", async (t) => {
  const f = fixture(t);
  const resources = [];
  function stuck() {
    const value = new EventEmitter();
    value.destroy = () => {
      value.destroyed = true;
    };
    resources.push(value);
    return value;
  }
  const doctor = networkDoctor({ get: stuck }, { createConnection: stuck });
  const { probe, ...dependencies } = f.dependencies;
  const report = await doctor.createDoctor({ ...dependencies, timeoutMs: 10 })(
    f.options,
  );
  await new Promise((resolve) => setTimeout(resolve, 20));
  assert.equal(resources.length, 7);
  assert.ok(resources.every(({ destroyed }) => destroyed));
  assert.equal(check(report, "service-public").code, "timeout");
});
