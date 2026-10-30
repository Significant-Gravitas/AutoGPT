"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { test } = require("node:test");

const {
  HELPER_ENTITLEMENTS,
  KEEP_VENDOR_SIGNATURE,
  RUNTIME_ENTITLEMENTS,
  SHELL_ENTITLEMENTS,
  codesignArguments,
  findCode,
  foreignTeams,
  ignoringRuntime,
  machOKind,
  parseSignature,
  planFor,
  requireNotarizedApp,
  runtimeRule,
  shellSignOptions,
  signatureProblem,
  unmatchedRules,
} = require("../build/mac_sign");

const DESKTOP = path.join(__dirname, "..");
const RESOURCES = path.join(DESKTOP, "resources");
const MH_EXECUTE = 2;
const MH_DYLIB = 6;

// What `codesign --display --verbose=2` prints.
const VENDOR = `Executable=/Applications/AutoGPT.app/Contents/Resources/runtime/site/claude_agent_sdk/_bundled/claude
Identifier=com.anthropic.claude-code
Format=Mach-O thin (arm64)
CodeDirectory v=20500 size=1404215 flags=0x10000(runtime) hashes=43871+7 location=embedded
Signature size=8986
Authority=Developer ID Application: Anthropic PBC (Q6L2SF6YDW)
Authority=Developer ID Certification Authority
Authority=Apple Root CA
Timestamp=Sep 24, 2026 at 18:02:11
Info.plist=not bound
TeamIdentifier=Q6L2SF6YDW
Runtime Version=15.5.0
Sealed Resources=none
Internal requirements count=1 size=180
`;
const ADHOC = `Executable=/tmp/runtime/python/bin/python3.13
Identifier=python3.13
Format=Mach-O thin (arm64)
CodeDirectory v=20400 size=52 flags=0x20002(adhoc,linker-signed) hashes=0+0 location=embedded
Signature=adhoc
Info.plist=not bound
TeamIdentifier=not set
Sealed Resources=none
Internal requirements=none
`;
const UNSIGNED = "/tmp/runtime/valkey/valkey-server: code object is not signed at all\n";

const vendor = parseSignature(VENDOR);
const adhoc = parseSignature(ADHOC);
const unsigned = parseSignature(UNSIGNED);

function temporaryDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "autogpt-sign-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function thinHeader(filetype) {
  const header = Buffer.alloc(32);
  header.writeUInt32LE(0xfeedfacf, 0);
  header.writeUInt32LE(filetype, 12);
  return header;
}

function universal(filetype) {
  const file = Buffer.alloc(4096 + 32);
  file.writeUInt32BE(0xcafebabe, 0);
  file.writeUInt32BE(2, 4);
  file.writeUInt32BE(4096, 16);
  thinHeader(filetype).copy(file, 4096);
  return file;
}

function write(dir, name, content) {
  const file = path.join(dir, ...name.split("/"));
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
  return file;
}

test("programs and libraries are told apart, and everything else is left out", (t) => {
  const dir = temporaryDir(t);
  assert.equal(machOKind(write(dir, "postgres", thinHeader(MH_EXECUTE))), "executable");
  assert.equal(machOKind(write(dir, "vector.dylib", thinHeader(MH_DYLIB))), "library");
  assert.equal(machOKind(write(dir, "universal", universal(MH_EXECUTE))), "executable");
  assert.equal(machOKind(write(dir, "universal.so", universal(MH_DYLIB))), "library");
  const javaClass = Buffer.alloc(64);
  javaClass.writeUInt32BE(0xcafebabe, 0);
  javaClass.writeUInt32BE(61, 4);
  assert.equal(machOKind(write(dir, "Main.class", javaClass)), null);
  assert.equal(machOKind(write(dir, "module.pyc", Buffer.from([0xf3, 0x0d, 0x0d, 0x0a, 0, 0, 0, 0]))), null);
  assert.equal(machOKind(write(dir, "empty", Buffer.alloc(0))), null);
  assert.equal(machOKind(write(dir, "rabbitmq-server", "#!/bin/sh\nexec erl\n")), null);
  assert.equal(machOKind(path.join(dir, "missing")), null);
});

test("only machine code is found, and links are not followed", (t) => {
  const dir = temporaryDir(t);
  write(dir, "python/bin/python3.13", thinHeader(MH_EXECUTE));
  write(dir, "site/numpy/core.so", thinHeader(MH_DYLIB));
  write(dir, "site/numpy/core.pyc", Buffer.from([0xf3, 0x0d, 0x0d, 0x0a]));
  const { code } = findCode(dir);
  assert.deepEqual(
    code.map((found) => [path.relative(dir, found.file).split(path.sep).join("/"), found.kind]),
    [
      ["python/bin/python3.13", "executable"],
      ["site/numpy/core.so", "library"],
    ],
  );
});

test("a link that points at nothing is reported, and one that points at a program is not signed twice", (t) => {
  const dir = temporaryDir(t);
  write(dir, "python/bin/python3.13", thinHeader(MH_EXECUTE));
  try {
    fs.symlinkSync("python3.13", path.join(dir, "python", "bin", "python3"));
    fs.symlinkSync("gone", path.join(dir, "python", "bin", "python3-config"));
  } catch (error) {
    if (error.code === "EPERM") return t.skip("this account may not create symbolic links");
    throw error;
  }
  const { code, dangling } = findCode(dir);
  assert.equal(code.length, 1);
  assert.deepEqual(dangling.map((file) => path.basename(file)), ["python3-config"]);
});

test("a signature is read for the four things notarization asks", () => {
  assert.deepEqual(vendor, {
    signed: true,
    adhoc: false,
    developerId: true,
    authority: "Developer ID Application: Anthropic PBC (Q6L2SF6YDW)",
    team: "Q6L2SF6YDW",
    hardened: true,
    timestamped: true,
  });
  assert.equal(signatureProblem(vendor), null);
  assert.equal(signatureProblem(adhoc), "ad-hoc signed");
  assert.equal(signatureProblem(unsigned), "not signed");
  assert.equal(signatureProblem({ ...vendor, hardened: false }), "no hardened runtime");
  assert.equal(signatureProblem({ ...vendor, timestamped: false }), "no secure timestamp");
  assert.match(
    signatureProblem(parseSignature(VENDOR.replace("Developer ID Application", "Apple Development"))),
    /not a Developer ID/,
  );
});

test("the Claude Code CLI keeps Anthropic's signature", () => {
  const relative = "site/claude_agent_sdk/_bundled/claude";
  assert.deepEqual(planFor({ relative, kind: "executable", signature: vendor }), { action: "keep" });
});

test("a Claude Code CLI that Apple would not notarize stops the build instead of being signed here", () => {
  const relative = "site/claude_agent_sdk/_bundled/claude";
  for (const signature of [adhoc, unsigned, { ...vendor, hardened: false }]) {
    const plan = planFor({ relative, kind: "executable", signature });
    assert.equal(plan.action, "refuse");
    assert.match(plan.reason, /ships unmodified/);
  }
});

test("anything else another vendor signed is signed again, so that one team's programs load it", () => {
  // Node, the BEAM and PostgreSQL validate libraries: a native addon or NIF
  // left under its vendor's Developer ID would be refused at dlopen.
  for (const [relative, kind] of [
    ["frontend/node_modules/@img/sharp-darwin-arm64/lib/sharp-darwin-arm64.node", "library"],
    ["erlang/lib/crypto-5.5/priv/lib/crypto.so", "library"],
    ["postgres/lib/vector.dylib", "library"],
    ["site/vendor/libthing.dylib", "library"],
    ["site/playwright/driver/node", "executable"],
  ]) {
    assert.deepEqual(planFor({ relative, kind, signature: vendor }), { action: "sign", entitlements: null });
  }
});

test("a file of another team is a problem unless it is one that keeps its vendor's signature", () => {
  const ours = { ...vendor, team: "ABCDE12345" };
  const files = [
    { name: "Contents/MacOS/AutoGPT", signature: ours, kept: false },
    { name: "Contents/Resources/runtime/site/claude_agent_sdk/_bundled/claude", signature: vendor, kept: true },
    { name: "Contents/Resources/runtime/node/addon.node", signature: vendor, kept: false },
    { name: "Contents/Resources/runtime/valkey/valkey-server", signature: unsigned, kept: false },
  ];
  const problems = foreignTeams(files, "ABCDE12345");
  assert.equal(problems.length, 1);
  assert.match(problems[0], /runtime\/node\/addon\.node: signed by team Q6L2SF6YDW, not by the app's \(ABCDE12345\)/);
  // An ad-hoc app has no team; its files are reported as ad-hoc signed instead.
  assert.deepEqual(foreignTeams(files, null), []);
});

test("each program that needs an entitlement gets its own, and is signed here whoever built it", () => {
  const expected = {
    "python/bin/python3.13": "entitlements.runtime.python.plist",
    "node/node": "entitlements.runtime.jit.plist",
    "erlang/erts-15.2.7/bin/beam.smp": "entitlements.runtime.jit.plist",
    "postgres/bin/postgres": "entitlements.runtime.dyld.plist",
    "postgres/bin/initdb": "entitlements.runtime.dyld.plist",
    "postgres/bin/pg_ctl": "entitlements.runtime.dyld.plist",
  };
  for (const [relative, plist] of Object.entries(expected)) {
    for (const signature of [adhoc, unsigned, vendor]) {
      assert.deepEqual(planFor({ relative, kind: "executable", signature }), {
        action: "sign",
        entitlements: path.join(RESOURCES, plist),
      });
    }
  }
});

test("libraries and every other program are signed without entitlements", () => {
  for (const [relative, kind] of [
    ["python/lib/libpython3.13.dylib", "library"],
    ["postgres/lib/vector.dylib", "library"],
    ["site/numpy/_core/_multiarray_umath.cpython-313-darwin.so", "library"],
    ["valkey/valkey-server", "executable"],
    ["prisma/query-engine-darwin-arm64", "executable"],
    ["erlang/erts-15.2.7/bin/epmd", "executable"],
  ]) {
    assert.deepEqual(planFor({ relative, kind, signature: adhoc }), { action: "sign", entitlements: null });
  }
  // A library in PostgreSQL's bin directory is not a program.
  assert.equal(planFor({ relative: "postgres/bin/libpq.5.dylib", kind: "library", signature: adhoc }).entitlements, null);
});

test("a rule that matches nothing is reported: the bundle's layout moved", () => {
  const layout = [
    "python/bin/python3.13",
    "node/node",
    "erlang/erts-15.2.7/bin/beam.smp",
    "postgres/bin/postgres",
    "site/claude_agent_sdk/_bundled/claude",
  ];
  assert.deepEqual(unmatchedRules(layout), []);
  const moved = layout.map((relative) => relative.replace("_bundled/claude", "bin/claude"));
  assert.equal(unmatchedRules(moved).length, 1);
  assert.match(unmatchedRules(moved)[0], /Claude Code CLI/);
  assert.equal(unmatchedRules([]).length, RUNTIME_ENTITLEMENTS.length + KEEP_VENDOR_SIGNATURE.length);
});

test("the rules name the paths the build puts these programs at", () => {
  const build = fs.readFileSync(path.join(DESKTOP, "build", "build_runtime.py"), "utf8");
  const layout = fs.readFileSync(path.join(DESKTOP, "runtime", "autogpt_desktop", "layout.py"), "utf8");
  const postgres = fs.readFileSync(path.join(DESKTOP, "runtime", "autogpt_desktop", "postgres.py"), "utf8");
  assert.ok(build.includes('"bin/python3"'), "build_runtime.py no longer puts Python at python/bin/python3");
  assert.ok(build.includes('out / "node" / f"node{EXE}"'), "build_runtime.py no longer puts Node at node/node");
  assert.ok(
    build.includes('"claude_agent_sdk" / "_bundled"'),
    "build_runtime.py no longer expects the Claude Code CLI in claude_agent_sdk/_bundled",
  );
  assert.ok(build.includes('target = self.out / "site"'), "third-party packages are no longer moved to site/");
  assert.ok(layout.includes('self.root / "node" / f"node{EXE}"'), "layout.py no longer runs node/node");
  assert.ok(
    postgres.includes("DYLD_FALLBACK_LIBRARY_PATH"),
    "postgres.py no longer sets DYLD_FALLBACK_LIBRARY_PATH: entitlements.runtime.dyld.plist can go",
  );
  assert.equal(runtimeRule("python/bin/python3.13").name, "python");
});

test("every entitlements file exists and grants exactly what is documented", () => {
  const keys = (file) =>
    [...fs.readFileSync(file, "utf8").matchAll(/<key>com\.apple\.security\.([^<]+)<\/key>/g)].map((match) => match[1]);
  const runtime = (name) => path.join(RESOURCES, name);
  assert.deepEqual(keys(SHELL_ENTITLEMENTS), ["cs.allow-jit", "device.audio-input"]);
  assert.deepEqual(keys(HELPER_ENTITLEMENTS), ["cs.allow-jit", "device.audio-input"]);
  assert.deepEqual(keys(runtime("entitlements.runtime.jit.plist")), ["cs.allow-jit"]);
  assert.deepEqual(keys(runtime("entitlements.runtime.dyld.plist")), ["cs.allow-dyld-environment-variables"]);
  assert.deepEqual(keys(runtime("entitlements.runtime.python.plist")), [
    "cs.allow-jit",
    "cs.allow-unsigned-executable-memory",
    "cs.disable-library-validation",
  ]);
  for (const rule of RUNTIME_ENTITLEMENTS) assert.ok(fs.existsSync(runtime(rule.plist)), rule.plist);
  for (const name of fs.readdirSync(RESOURCES).filter((file) => file.endsWith(".plist"))) {
    const text = fs.readFileSync(runtime(name), "utf8");
    // Nothing that would let a debugger attach, which notarization refuses.
    assert.ok(!text.includes("get-task-allow"), name);
    // codesign's parser has rejected comments in an entitlements file.
    assert.ok(!text.includes("<!--"), name);
    assert.ok(!text.includes("\r"), `${name} has CRLF line endings`);
  }
});

test("electron-builder does not pick the entitlements up by itself for the unsigned build", () => {
  // It uses resources/entitlements.mac.plist and .mac.inherit.plist when
  // they exist, signed or not.
  for (const name of ["entitlements.mac.plist", "entitlements.mac.inherit.plist"]) {
    assert.ok(!fs.existsSync(path.join(RESOURCES, name)), name);
  }
});

test("codesign is asked for a Developer ID signature, a timestamp and the hardened runtime", () => {
  assert.deepEqual(codesignArguments({ identity: "ABCDEF0123", keychain: null, entitlements: null }), [
    "--sign",
    "ABCDEF0123",
    "--force",
    "--timestamp",
    "--options",
    "runtime",
  ]);
  const full = codesignArguments({ identity: "ABCDEF0123", keychain: "/tmp/build.keychain", entitlements: "/e.plist" });
  assert.deepEqual(full.slice(-4), ["--keychain", "/tmp/build.keychain", "--entitlements", "/e.plist"]);
});

test("Electron's signer is told to leave the runtime alone, with a function it does not drop", () => {
  const app = path.join(path.sep, "dist", "mac-arm64", "AutoGPT.app");
  const inside = path.join(app, "Contents", "Resources", "runtime", "site", "x.so");
  const framework = path.join(app, "Contents", "Frameworks", "Electron Framework.framework");
  const kext = path.join(app, "Contents", "Resources", "driver.kext");

  const ignore = ignoringRuntime(app, (file) => file.endsWith(".kext"));
  assert.equal(typeof ignore, "function");
  assert.equal(ignore(inside), true);
  assert.equal(ignore(kext), true);
  assert.equal(ignore(framework), false);
  assert.equal(ignore(path.join(app, "Contents", "Resources", "runtime-notes.txt")), false);
  assert.equal(ignoringRuntime(app, undefined)(framework), false);
  assert.equal(ignoringRuntime(app, [/Frameworks/])(framework), true);
});

test("@electron/osx-sign still takes a function for `ignore` and still exports signApp", (t) => {
  const file = path.join(DESKTOP, "node_modules", "@electron", "osx-sign", "dist", "cjs", "sign.js");
  if (!fs.existsSync(file)) return t.skip("@electron/osx-sign is not installed (npm ci)");
  const source = fs.readFileSync(file, "utf8");
  assert.ok(source.includes("typeof ignore === 'function'"), "osx-sign no longer accepts a function for `ignore`");
  assert.ok(source.includes("exports.signApp"), "osx-sign no longer exports signApp");
});

test("the shell is signed with the options electron-builder would use", () => {
  const app = path.join(path.sep, "dist", "AutoGPT.app");
  const options = shellSignOptions({ app, identity: "ABCDEF0123", keychain: undefined });
  assert.equal(options.identityValidation, false);
  assert.equal(options.platform, "darwin");
  assert.deepEqual(options.optionsForFile(app), { entitlements: SHELL_ENTITLEMENTS, hardenedRuntime: true });
  assert.deepEqual(options.optionsForFile(path.join(app, "Contents", "Frameworks", "AutoGPT Helper.app")), {
    entitlements: HELPER_ENTITLEMENTS,
    hardenedRuntime: true,
  });
});

// electron-builder's answers to a Developer ID build, in what it leaves on disk.
function codesignSays(display, stapled = true) {
  const calls = [];
  const execute = async (command, args) => {
    calls.push([command, ...args].join(" "));
    if (command === "codesign") return { ok: true, output: display };
    return { ok: stapled, output: stapled ? "The validate action worked!" : "AutoGPT.app does not have a ticket stapled to it." };
  };
  return { calls, execute };
}

test("a build that was asked to sign fails when electron-builder left the app unsigned", async () => {
  const app = "/dist/mac-arm64/AutoGPT.app";
  for (const [display, problem] of [
    [`${app}: code object is not signed at all\n`, /AutoGPT\.app is not signed/],
    [ADHOC, /AutoGPT\.app is ad-hoc signed/],
    [VENDOR.replace("Developer ID Application", "Apple Development"), /not a Developer ID/],
  ]) {
    const { calls, execute } = codesignSays(display);
    await assert.rejects(requireNotarizedApp(app, execute), problem);
    assert.deepEqual(calls, [`codesign --display --verbose=2 ${app}`]);
  }
});

test("a build that was asked to sign fails when the app was not notarized", async () => {
  const app = "/dist/mac-arm64/AutoGPT.app";
  const skipped = codesignSays(VENDOR, false);
  await assert.rejects(requireNotarizedApp(app, skipped.execute), /no notarization ticket.*does not have a ticket/s);
  const done = codesignSays(VENDOR, true);
  await requireNotarizedApp(app, done.execute);
  assert.deepEqual(done.calls, [`codesign --display --verbose=2 ${app}`, `xcrun stapler validate ${app}`]);
});

test("electron-builder still packs an unsigned app when a custom signer finds no identity", (t) => {
  // The reason for requireNotarizedApp. If electron-builder starts failing
  // by itself here (or runs afterSign regardless), the hook can go.
  const lib = path.join(DESKTOP, "node_modules", "app-builder-lib", "out");
  if (!fs.existsSync(lib)) return t.skip("electron-builder is not installed (npm ci)");
  const helper = fs.readFileSync(path.join(lib, "mac", "MacTargetHelper.js"), "utf8");
  assert.ok(helper.includes("const noIdentity = !config.sign && identity == null;"), "findSigningIdentity changed");
  const packager = fs.readFileSync(path.join(lib, "packager.js"), "utf8");
  assert.ok(packager.includes('emit("artifactBuildStarted", event)'), "artifactBuildStarted is no longer awaited");
  const dmg = fs.readFileSync(path.join(DESKTOP, "node_modules", "dmg-builder", "out", "dmg.js"), "utf8");
  assert.ok(dmg.includes("await packager.info.emitArtifactBuildStarted("), "the disk image no longer announces itself");
});
