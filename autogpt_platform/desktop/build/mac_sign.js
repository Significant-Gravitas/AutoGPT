"use strict";

// Developer ID signing of the macOS app, for notarization.
//
//   node build/mac_sign.js sign  <AutoGPT.app> --identity <SHA-1> [--keychain <file>]
//   node build/mac_sign.js check <AutoGPT.app>
//
// electron-builder calls signMacApp (electron-builder.config.js, `mac.sign`)
// when a certificate is present; build/sign_check_macos.sh calls the same
// code by hand. Without a certificate none of this runs and the app is
// ad-hoc signed by electron-builder as before.
//
// Apple notarizes an app only if every Mach-O file in it carries a Developer
// ID signature with a secure timestamp and the hardened runtime. The app has
// two parts, signed differently:
//
// 1. The runtime (Contents/Resources/runtime): CPython and its extension
//    modules, Node, PostgreSQL, Erlang, Valkey, the Prisma engines and the
//    Claude Code CLI. It is signed here, file by file and in parallel.
//    @electron/osx-sign cannot do it: it signs every file that is not text,
//    one at a time (20,000 bytecode and .beam files), and gives all of them
//    one entitlements file.
//
//    Every file gets this build's signature, whoever built it, with one
//    exception: the programs in KEEP_VENDOR_SIGNATURE. The Claude Code CLI
//    is Anthropic's signed program and must ship unmodified; signing it
//    again would change it. Notarization accepts another vendor's Developer
//    ID inside the app. Nothing else keeps a signature it came with: under
//    the hardened runtime a program loads only libraries signed by its own
//    team, so a library left under its vendor's signature would be refused
//    by Node, the BEAM and PostgreSQL on a user's Mac, where no test runs.
//
//    Entitlements belong to an executable and are not passed on to the
//    programs it starts, so each program that needs one gets its own
//    (RUNTIME_ENTITLEMENTS). Libraries and every other executable get none.
//
// 2. Electron itself (everything else in the bundle), signed by
//    @electron/osx-sign with the same options electron-builder gives it,
//    told to leave the runtime alone. It signs the bundle last, which seals
//    the runtime's files as resources.

const { execFile } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const RESOURCES = path.join(__dirname, "..", "resources");
const RUNTIME = ["Contents", "Resources", "runtime"];
const SHELL_ENTITLEMENTS = path.join(RESOURCES, "entitlements.shell.plist");
const HELPER_ENTITLEMENTS = path.join(RESOURCES, "entitlements.shell.helper.plist");

// Paths inside the runtime, with forward slashes (the layout is
// build/build_runtime.py's). The first match wins. Every rule must match a
// file, so a layout change fails the build here and not at run time on a
// user's Mac.
const RUNTIME_ENTITLEMENTS = [
  {
    // ctypes/cffi callbacks and JIT-compiling packages make memory
    // executable, and packages load libraries they build or unpack while
    // running, which no team has signed.
    name: "python",
    match: /^python\/bin\/python3(\.\d+)?$/,
    plist: "entitlements.runtime.python.plist",
  },
  {
    // V8 compiles JavaScript into MAP_JIT memory.
    name: "node",
    match: /^node\/node$/,
    plist: "entitlements.runtime.jit.plist",
  },
  {
    // The BEAM's JIT (BeamAsm) does the same for Erlang code.
    name: "erlang",
    match: /^erlang\/(.+\/)?bin\/beam\.smp$/,
    plist: "entitlements.runtime.jit.plist",
  },
  {
    // runtime/autogpt_desktop/postgres.py points PostgreSQL's programs at
    // their libraries with DYLD_FALLBACK_LIBRARY_PATH, which the hardened
    // runtime removes from the environment of a program without this.
    name: "postgres",
    match: /^postgres\/bin\/[^/]+$/,
    plist: "entitlements.runtime.dyld.plist",
  },
];

// Programs that must keep their vendor's signature: self-contained ones,
// which load no library from the bundle. If one of them is not signed in a
// way Apple will notarize, the build stops: signing it here is not an option.
const KEEP_VENDOR_SIGNATURE = [
  { name: "the Claude Code CLI", match: /^site\/claude_agent_sdk\/_bundled\/claude$/ },
];

const MH_EXECUTE = 2;
const FAT_MAGIC = 0xcafebabe;
// A Java class file starts with the same four bytes as a universal binary;
// what follows is its version (45 or more) where a binary has a small count.
const MAX_FAT_ARCHITECTURES = 30;
const SIGNERS = 8;
const SIGN_ATTEMPTS = 3;

// "executable", "library" (anything else that holds machine code) or null.
function machOKind(file) {
  let fd;
  try {
    fd = fs.openSync(file, "r");
    return kindAt(fd, 0, true);
  } catch {
    return null;
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
}

function kindAt(fd, offset, mayBeFat) {
  const header = Buffer.alloc(20);
  if (fs.readSync(fd, header, 0, header.length, offset) < header.length) return null;
  const magic = header.readUInt32BE(0);
  if (magic === FAT_MAGIC) {
    if (!mayBeFat || header.readUInt32BE(4) > MAX_FAT_ARCHITECTURES) return null;
    // fat_header (8 bytes), then fat_arch: cputype, cpusubtype, offset.
    return kindAt(fd, header.readUInt32BE(16), false);
  }
  if (magic === 0xfeedface || magic === 0xfeedfacf) {
    return header.readUInt32BE(12) === MH_EXECUTE ? "executable" : "library";
  }
  if (magic === 0xcefaedfe || magic === 0xcffaedfe) {
    return header.readUInt32LE(12) === MH_EXECUTE ? "executable" : "library";
  }
  return null;
}

// Every Mach-O file under `root`, and every symbolic link that points at
// nothing. Links are not followed: what they point at is found by itself.
function findCode(root) {
  const code = [];
  const dangling = [];
  const pending = [root];
  while (pending.length > 0) {
    const dir = pending.pop();
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        if (!fs.existsSync(file)) dangling.push(file);
      } else if (entry.isDirectory()) {
        pending.push(file);
      } else if (entry.isFile()) {
        const kind = machOKind(file);
        if (kind) code.push({ file, kind });
      }
    }
  }
  code.sort((a, b) => a.file.localeCompare(b.file));
  return { code, dangling: dangling.sort() };
}

// What `codesign --display --verbose=2` printed about a file.
function parseSignature(text) {
  const authority = /^Authority=(.+)$/m.exec(text)?.[1] || null;
  const team = /^TeamIdentifier=(.+)$/m.exec(text)?.[1] || null;
  const flags = /^CodeDirectory .*flags=\S+?\(([^)]*)\)/m.exec(text)?.[1].split(",") || [];
  return {
    signed: /^CodeDirectory /m.test(text),
    adhoc: /^Signature=adhoc$/m.test(text) || flags.includes("adhoc"),
    developerId: Boolean(authority?.startsWith("Developer ID Application:")),
    authority,
    team: team === "not set" ? null : team,
    hardened: flags.includes("runtime"),
    timestamped: /^Timestamp=/m.test(text),
  };
}

// The four things Apple's notary service asks of every Mach-O file.
function signatureProblem(signature) {
  if (!signature.signed) return "not signed";
  if (signature.adhoc) return "ad-hoc signed";
  if (!signature.developerId) return `signed by ${signature.authority || "an unknown authority"}, not a Developer ID`;
  if (!signature.hardened) return "no hardened runtime";
  if (!signature.timestamped) return "no secure timestamp";
  return null;
}

function runtimeRule(relative) {
  return RUNTIME_ENTITLEMENTS.find((rule) => rule.match.test(relative)) || null;
}

function mustKeep(relative) {
  return KEEP_VENDOR_SIGNATURE.find((rule) => rule.match.test(relative)) || null;
}

// What to do with one Mach-O file of the runtime:
//   { action: "keep" }                 leave the vendor's signature
//   { action: "sign", entitlements }   sign it (entitlements: a file or null)
//   { action: "refuse", reason }       it must be kept and cannot be
function planFor({ relative, kind, signature }) {
  const kept = mustKeep(relative);
  if (kept) {
    const problem = signatureProblem(signature);
    if (!problem) return { action: "keep" };
    return {
      action: "refuse",
      reason: `${kept.name} (${relative}) is ${problem}. It ships unmodified, so it cannot be signed here: bundle a build its vendor signed for notarization.`,
    };
  }
  // Everything else is signed here, also what its vendor signed properly
  // (see the top of this file).
  const rule = kind === "executable" ? runtimeRule(relative) : null;
  return { action: "sign", entitlements: rule ? path.join(RESOURCES, rule.plist) : null };
}

// Rules that matched nothing: the bundle's layout moved under them.
function unmatchedRules(relatives) {
  return [...RUNTIME_ENTITLEMENTS, ...KEEP_VENDOR_SIGNATURE]
    .filter((rule) => !relatives.some((relative) => rule.match.test(relative)))
    .map((rule) => `${rule.name} (${rule.match})`);
}

function codesignArguments({ identity, keychain, entitlements }) {
  const args = ["--sign", identity, "--force", "--timestamp", "--options", "runtime"];
  if (keychain) args.push("--keychain", keychain);
  if (entitlements) args.push("--entitlements", entitlements);
  return args;
}

// The same programs as regular expressions on a full path, for
// electron-builder's `signIgnore`: the build without a certificate signs
// everything ad hoc, which would replace their signature just the same.
function vendorSignedPaths() {
  return KEEP_VENDOR_SIGNATURE.map(
    (rule) => `/${RUNTIME.join("/")}/${rule.match.source.replace(/^\^/, "")}`,
  );
}

function runtimeDir(app) {
  return path.join(app, ...RUNTIME);
}

function inRuntime(app, file) {
  return file.startsWith(runtimeDir(app) + path.sep);
}

function relativeTo(root, file) {
  return path.relative(root, file).split(path.sep).join("/");
}

function run(command, args) {
  return new Promise((resolve) => {
    execFile(command, args, { maxBuffer: 16 * 1024 * 1024 }, (error, stdout, stderr) => {
      resolve({ ok: !error, output: `${stdout}${stderr}` });
    });
  });
}

async function inParallel(items, limit, work) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await work(items[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

async function readSignature(file) {
  return parseSignature((await run("codesign", ["--display", "--verbose=2", file])).output);
}

// Apple's timestamp server refuses a request now and then.
async function codesign(file, options) {
  let last;
  for (let attempt = 1; attempt <= SIGN_ATTEMPTS; attempt += 1) {
    last = await run("codesign", [...codesignArguments(options), file]);
    if (last.ok) return null;
    await new Promise((resolve) => setTimeout(resolve, attempt * 2000));
  }
  return `${file}: ${last.output.trim()}`;
}

function fail(title, lines) {
  throw new Error(`${title}:\n  ${lines.join("\n  ")}`);
}

// Signs the runtime's Mach-O files. Returns what was kept, for the log.
async function signRuntime({ app, identity, keychain, log = console.log }) {
  const root = runtimeDir(app);
  const { code, dangling } = findCode(root);
  // @electron/osx-sign follows links as it walks the bundle and stops at one
  // that leads nowhere.
  if (dangling.length > 0) fail("Symbolic links in the runtime that point at nothing", dangling);
  const files = code.map((found) => ({ ...found, relative: relativeTo(root, found.file) }));
  const unmatched = unmatchedRules(files.map((found) => found.relative));
  if (unmatched.length > 0) {
    fail("The runtime's layout changed; update the rules in build/mac_sign.js. Nothing matches", unmatched);
  }

  const signatures = await inParallel(files, SIGNERS, (found) => readSignature(found.file));
  const plans = files.map((found, index) => ({
    ...found,
    signature: signatures[index],
    ...planFor({ ...found, signature: signatures[index] }),
  }));
  const refused = plans.filter((plan) => plan.action === "refuse");
  if (refused.length > 0) fail("Cannot sign the runtime", refused.map((plan) => plan.reason));

  const toSign = plans.filter((plan) => plan.action === "sign");
  log(`Signing ${toSign.length} of the runtime's ${plans.length} Mach-O files`);
  const failures = (
    await inParallel(toSign, SIGNERS, (plan) =>
      codesign(plan.file, { identity, keychain, entitlements: plan.entitlements }),
    )
  ).filter(Boolean);
  if (failures.length > 0) fail(`codesign failed for ${failures.length} files`, failures);

  const kept = plans.filter((plan) => plan.action === "keep");
  for (const plan of kept) {
    log(`Kept the signature of ${plan.signature.authority} on ${plan.kind} ${plan.relative}`);
  }
  return { signed: toSign.length, kept: kept.map((plan) => plan.relative) };
}

// What electron-builder passes to @electron/osx-sign for a Developer ID
// build (app-builder-lib, MacTargetHelper.buildSignOptions), for signing
// without electron-builder.
function shellSignOptions({ app, identity, keychain }) {
  return {
    app,
    identity,
    identityValidation: false,
    keychain,
    platform: "darwin",
    type: "distribution",
    optionsForFile: (file) => ({
      entitlements: file === app ? SHELL_ENTITLEMENTS : HELPER_ENTITLEMENTS,
      hardenedRuntime: true,
    }),
  };
}

// `mac.sign` in electron-builder.config.js. `options` are electron-builder's
// for @electron/osx-sign; its own `ignore` is kept. electron-builder calls
// this only when it has found an identity: see requireNotarizedApp.
async function signMacApp(options) {
  await signRuntime({ app: options.app, identity: options.identity, keychain: options.keychain });
  const { signApp } = require("@electron/osx-sign");
  await signApp({ ...options, ignore: ignoringRuntime(options.app, options.ignore) });
}

// One function, not a list: @electron/osx-sign 1.3.3 drops an `ignore` that
// is an array (validateOptsIgnore returns nothing for one).
function ignoringRuntime(app, previous) {
  const others = [].concat(previous || []);
  const ignored = (rule, file) => (typeof rule === "function" ? rule(file) : Boolean(file.match(rule)));
  return (file) => inRuntime(app, file) || others.some((rule) => ignored(rule, file));
}

// `artifactBuildStarted` in electron-builder.config.js, for a build that was
// asked for a Developer ID signature. electron-builder 26 neither signs nor
// fails when `mac.sign` is a function and it finds no Developer ID
// Application identity (MacTargetHelper.findSigningIdentity; it does not
// consult forceCodeSigning then, and skips `afterSign` too), and it skips
// notarization with a warning when the App Store Connect key is missing.
// Either would end in a disk image of an app that every Mac refuses, so the
// packed app is looked at before any installer is made from it.
async function requireNotarizedApp(app, execute = run) {
  const name = path.basename(app);
  const signature = parseSignature((await execute("codesign", ["--display", "--verbose=2", app])).output);
  const problem = signatureProblem(signature);
  if (problem) {
    throw new Error(
      `A Developer ID build was asked for, but ${name} is ${problem}. electron-builder signs only with a valid ` +
        "'Developer ID Application' certificate: look at CSC_LINK, or at the keychain with " +
        "`security find-identity -v -p codesigning`.",
    );
  }
  const stapled = await execute("xcrun", ["stapler", "validate", app]);
  if (!stapled.ok) {
    throw new Error(
      `A Developer ID build was asked for, but ${name} carries no notarization ticket. Notarization needs ` +
        `APPLE_API_KEY, APPLE_API_KEY_ID and APPLE_API_ISSUER. stapler says: ${stapled.output.trim()}`,
    );
  }
}

// Files signed by a team other than the app's own. Only the programs in
// KEEP_VENDOR_SIGNATURE may be: a library of another team is refused by
// every program that validates libraries, which is all of them but Python.
function foreignTeams(files, appTeam) {
  if (!appTeam) return [];
  return files
    .filter((file) => !file.kept && file.signature.team && file.signature.team !== appTeam)
    .map((file) => `${file.name}: signed by team ${file.signature.team}, not by the app's (${appTeam})`);
}

// Everything that would make Apple refuse the app, or make it fail on a
// user's Mac: each Mach-O file's signature and team, and the entitlements of
// the programs that need one.
async function checkApp(app) {
  const { code } = findCode(app);
  const signatures = await inParallel(code, SIGNERS, (found) => readSignature(found.file));
  const problems = [];
  code.forEach((found, index) => {
    const problem = signatureProblem(signatures[index]);
    if (problem) problems.push(`${relativeTo(app, found.file)}: ${problem}`);
  });
  const files = code.map((found, index) => ({
    name: relativeTo(app, found.file),
    signature: signatures[index],
    kept: inRuntime(app, found.file) && Boolean(mustKeep(relativeTo(runtimeDir(app), found.file))),
  }));
  problems.push(...foreignTeams(files, (await readSignature(app)).team));

  const root = runtimeDir(app);
  const runtime = code.filter((found) => inRuntime(app, found.file));
  const relatives = runtime.map((found) => relativeTo(root, found.file));
  for (const rule of unmatchedRules(relatives)) problems.push(`nothing matches the rule ${rule}`);
  for (const [index, found] of runtime.entries()) {
    const rule = found.kind === "executable" ? runtimeRule(relatives[index]) : null;
    if (!rule || mustKeep(relatives[index])) continue;
    const granted = (await run("codesign", ["--display", "--entitlements", "-", "--xml", found.file])).output;
    for (const key of entitlementKeys(path.join(RESOURCES, rule.plist))) {
      if (!granted.includes(key)) problems.push(`runtime/${relatives[index]}: lacks the entitlement ${key}`);
    }
  }
  return { files: code.length, teams: countTeams(signatures), problems };
}

function entitlementKeys(plist) {
  return [...fs.readFileSync(plist, "utf8").matchAll(/<key>([^<]+)<\/key>/g)].map((match) => match[1]);
}

function countTeams(signatures) {
  const teams = {};
  for (const signature of signatures) {
    const name = signature.developerId ? signature.authority : signature.adhoc ? "ad-hoc" : "not signed";
    teams[name] = (teams[name] || 0) + 1;
  }
  return teams;
}

function option(args, name) {
  const index = args.indexOf(name);
  return index === -1 ? undefined : args[index + 1];
}

async function main([command, app, ...rest]) {
  if (!app || !["sign", "check"].includes(command)) {
    console.error("usage: mac_sign.js sign <App.app> --identity <SHA-1> [--keychain <file>]\n       mac_sign.js check <App.app>");
    return 2;
  }
  const target = path.resolve(app);
  if (command === "sign") {
    const identity = option(rest, "--identity");
    if (!identity) throw new Error("--identity is required: the SHA-1 hash of the Developer ID Application certificate");
    await signMacApp(shellSignOptions({ app: target, identity, keychain: option(rest, "--keychain") }));
    return 0;
  }
  const { files, teams, problems } = await checkApp(target);
  console.log(`${files} Mach-O files`);
  for (const [name, count] of Object.entries(teams)) console.log(`  ${count}  ${name}`);
  for (const problem of problems) console.log(`PROBLEM  ${problem}`);
  console.log(problems.length === 0 ? "Every Mach-O file is ready for notarization." : `${problems.length} problems.`);
  return problems.length === 0 ? 0 : 1;
}

if (require.main === module) {
  main(process.argv.slice(2)).then(
    (code) => process.exit(code),
    (error) => {
      console.error(error.message);
      process.exit(1);
    },
  );
}

module.exports = {
  HELPER_ENTITLEMENTS,
  KEEP_VENDOR_SIGNATURE,
  RUNTIME_ENTITLEMENTS,
  SHELL_ENTITLEMENTS,
  checkApp,
  codesignArguments,
  findCode,
  foreignTeams,
  ignoringRuntime,
  inRuntime,
  machOKind,
  parseSignature,
  planFor,
  requireNotarizedApp,
  runtimeRule,
  shellSignOptions,
  signMacApp,
  signatureProblem,
  unmatchedRules,
  vendorSignedPaths,
};
