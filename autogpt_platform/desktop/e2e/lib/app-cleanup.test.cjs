const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const ts = require("typescript");

const source = fs.readFileSync(path.join(__dirname, "app.ts"), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;

function fixture(found, inventoryError) {
  const signals = [];
  const logs = [];
  const alive = new Set(found.map(({ pid }) => pid));
  const exports = {};
  const dependencies = {
    "node:fs": fs,
    "node:path": path,
    "@playwright/test": {},
    "./attach": {},
    "./durations": {},
    "./config": { dataDir: () => "synthetic-test-data" },
    "./firewall": {},
    "./platform": {
      appProcesses() {
        if (inventoryError) throw inventoryError;
        return found;
      },
    },
    "./processes": {
      describe: (items) => items.map(({ pid, executable }) => `${pid} ${executable}`).join("\n"),
    },
    "./runtime-log": { logTails: () => "SYNTHETIC_RUNTIME_LOG" },
    "./snapshot": {},
  };
  vm.runInNewContext(compiled, {
    exports,
    require(name) {
      assert.ok(Object.hasOwn(dependencies, name), `unexpected dependency: ${name}`);
      return dependencies[name];
    },
    process: {
      kill(pid, signal) {
        signals.push({ pid, signal });
        alive.delete(pid);
      },
    },
    console: { log: (message) => logs.push(message) },
  }, { filename: "app.ts" });
  return { stopStrayApp: exports.stopStrayApp, signals, logs, alive };
}

test("failure cleanup leaves borrowed runtime and presumed app processes untouched", async () => {
  const found = [
    {
      pid: 4242,
      executable: "C:\\owned-install\\resources\\runtime\\postgres\\bin\\postgres.exe",
      command: "postgres.exe -D C:\\unrelated-project\\database",
    },
    {
      pid: 4343,
      executable: "C:\\owned-install\\AutoGPT.exe",
      command: "AutoGPT.exe",
    },
  ];
  const check = fixture(found);
  const failure = await check.stopStrayApp().then(() => null, (error) => error);
  assert.deepEqual(check.signals, [], "an executable path is not authority to terminate a process");
  assert.deepEqual([...check.alive], [4242, 4343]);
  assert.match(failure?.message || "", /ownership/i);
  assert.match(failure.message, /manual/i);
  const report = check.logs.join("\n");
  assert.match(report, /4242.*postgres\.exe/);
  assert.match(report, /4343.*AutoGPT\.exe/);
  assert.match(report, /SYNTHETIC_RUNTIME_LOG/);
});

test("failure cleanup succeeds quietly when no matching process remains", async () => {
  const check = fixture([]);
  await check.stopStrayApp();
  assert.deepEqual(check.signals, []);
  assert.deepEqual(check.logs, []);
});

test("failure cleanup preserves a process-inventory error without terminating anything", async () => {
  const check = fixture([], new Error("cannot inspect processes"));
  await assert.rejects(check.stopStrayApp(), /cannot inspect processes/);
  assert.deepEqual(check.signals, []);
});
