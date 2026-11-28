const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { PassThrough } = require("node:stream");
const test = require("node:test");
const vm = require("node:vm");
const ts = require("typescript");

function load(name, dependencies = {}, globals = {}) {
  const source = fs.readFileSync(path.join(__dirname, name), "utf8");
  const compiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true,
    },
  }).outputText;
  const exports = {};
  vm.runInNewContext(compiled, {
    exports, URL, AbortSignal, fetch, setTimeout, clearTimeout, process,
    require(name) {
      if (Object.hasOwn(dependencies, name)) return dependencies[name];
      assert.ok(name.startsWith("node:"), `unexpected dependency: ${name}`);
      return require(name);
    },
    ...globals,
  }, { filename: name });
  return exports;
}

test("a non-answering debugging endpoint cannot outlive the attach deadline", { timeout: 15000 }, async (t) => {
  const server = http.createServer(() => {});
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  let connected = false;
  const { attach } = load("attach.ts", {
    "@playwright/test": { chromium: { connectOverCDP() { connected = true; } } },
  });
  await assert.rejects(attach(server.address().port, "http://localhost:3000", 100), /timed out/);
  assert.equal(connected, false);
});

test("CDP connection receives only the remaining attachment budget", async () => {
  let now = 1000;
  let options;
  const page = { url: () => "http://localhost:3000/settings" };
  const browser = { contexts: () => [{ pages: () => [page] }] };
  const { attach } = load("attach.ts", {
    "@playwright/test": { chromium: { async connectOverCDP(_endpoint, value) { options = value; return browser; } } },
  }, {
    Date: { now: () => now },
    fetch: async () => { now += 40; return { ok: true }; },
  });
  const attached = await attach(9222, "http://localhost:3000", 100);
  assert.equal(options.timeout, 60);
  assert.equal(attached.page, page);
});

function windowsPlatform(processes, spawn) {
  return load("platform.ts", {
    "node:path": path.win32,
    "node:fs": { existsSync: () => false },
    "node:child_process": { spawn },
    "./config": {
      kind: "nsis", installDirOverride: "C:\\owned install\\日本",
      product: { packageName: "autogpt", productFilename: "AutoGPT" },
    },
    "./processes": processes,
  }, { process: { platform: "win32", env: {} } });
}

test("NSIS keeps the final directory argument unquoted and quotes argv0", async () => {
  const calls = [];
  function spawn(command, args, options) {
    calls.push({ command, args, options });
    const child = new EventEmitter();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    queueMicrotask(() => child.emit("close", 0));
    return child;
  }
  const { platform, run } = windowsPlatform({}, spawn);
  const installer = "C:\\installer files\\AutoGPT Setup.exe";
  await platform.install(installer);
  assert.equal(calls[0].command, installer);
  assert.deepEqual(Array.from(calls[0].args), ["/S", "/D=C:\\owned install\\日本"]);
  assert.equal(calls[0].options.windowsVerbatimArguments, true);
  assert.equal(calls[0].options.argv0, `"${installer}"`);
  await run("ordinary-tool", ["an argument with spaces"]);
  assert.equal(calls[1].options.windowsVerbatimArguments, false);
  assert.equal(calls[1].options.argv0, undefined);
});

test("Windows inventory preserves Unicode paths without changing console encoding", {
  skip: process.platform !== "win32",
  timeout: 30000,
}, () => {
  const executable = "C:\\owned install\\日本\\AutoGPT.exe";
  let checkedEncoding = false;
  const processes = load("processes.ts", {
    "node:path": path.win32,
    "node:child_process": {
      execFileSync(command, args, options) {
        const script = args.at(-1);
        // A synthetic CIM boundary keeps unrelated host process data out of the test.
        const fixture = Buffer.from(JSON.stringify({
          ProcessId: 4242, ExecutablePath: executable, CommandLine: `${executable} --fixture`,
        })).toString("base64");
        const prefix = `$originalEncoding = [Console]::OutputEncoding; ` +
          `try { [Console]::OutputEncoding = [Text.Encoding]::GetEncoding(437); ` +
          `$before = [Console]::OutputEncoding.CodePage; ` +
          `function Get-CimInstance { [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${fixture}')) | ConvertFrom-Json }; `;
        const suffix = `; if ([Console]::OutputEncoding.CodePage -ne $before) { throw 'console encoding changed' } ` +
          `} finally { [Console]::OutputEncoding = $originalEncoding }`;
        const output = execFileSync(command, [...args.slice(0, -1), prefix + script + suffix], {
          ...options, windowsHide: true, timeout: 20000,
        });
        assert.equal(JSON.parse(output)[0]?.ExecutablePath, executable, output);
        checkedEncoding = true;
        return output;
      },
    },
  }, { process: { platform: "win32" } });
  const { appProcesses } = windowsPlatform(processes);
  const found = appProcesses();
  assert.equal(found.length, 1);
  assert.equal(found[0].pid, 4242);
  assert.equal(found[0].executable, executable);
  assert.equal(found[0].command, `${executable} --fixture`);
  assert.equal(checkedEncoding, true);
});
