#!/usr/bin/env node
// Orca worktree hooks for AutoGPT, invoked from orca.yaml as
// `node .orca/hooks.mjs <setup|archive>`. Node is the entry point (rather than
// a shell script) because Orca runs hooks under /bin/bash on macOS/Linux but
// cmd.exe on Windows, and `node` resolves identically in both.
import {
  chmodSync,
  copyFileSync,
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const mode = process.argv[2];
const worktree = process.env.ORCA_WORKTREE_PATH ?? process.cwd();
const root = process.env.ORCA_ROOT_PATH;

// Every directory that carries .env* files. When a new service with its own
// .env joins the monorepo, add its directory here — otherwise its env is
// silently not linked into worktrees and not captured by archive().
const ENV_DIRS = [
  "",
  "autogpt_platform",
  "autogpt_platform/backend",
  "autogpt_platform/frontend",
  "autogpt_platform/db/docker",
];

function run(cmd, cwd) {
  if (cwd && !existsSync(cwd)) {
    // Older branches predate some of these directories; a missing one means
    // there is nothing to install, not a failure worth aborting setup over.
    console.log(`skipping (no such directory: ${cwd}): ${cmd}`);
    return;
  }
  console.log(`$ ${cmd}  (in ${cwd})`);
  const r = spawnSync(cmd, { shell: true, stdio: "inherit", cwd });
  if (r.status !== 0) {
    console.error(`command failed with status ${r.status}: ${cmd}`);
    process.exit(r.status ?? 1);
  }
}

function git(args, cwd, input) {
  // maxBuffer: worktrees can carry very large uncommitted diffs; the 1MB
  // spawnSync default would silently truncate exactly what archive() backs up
  return spawnSync("git", args, {
    encoding: "utf8",
    cwd,
    input,
    maxBuffer: 1024 ** 3,
  });
}

function gitStrict(args, cwd) {
  const r = git(args, cwd);
  if (r.error || r.status !== 0) {
    console.error(
      `git ${args.join(" ")} failed: ${r.error?.message ?? r.stderr}`,
    );
    process.exit(1);
  }
  return r.stdout ?? "";
}

// Same, but returning raw stdout bytes. Patch output must never be decoded:
// git base85-encodes only the blobs it classifies as *binary* (i.e. containing
// a NUL byte), so a tracked text file that is merely not valid UTF-8 — latin-1
// fixtures, legacy .po/.csv data — travels through `git diff --binary` as raw
// bytes. Decoding those to a JS string turns them into U+FFFD, the patch stops
// matching the blob it came from, and `git apply` rejects the developer's only
// surviving copy of the change.
function gitStrictRaw(args, cwd) {
  const r = spawnSync("git", args, { cwd, maxBuffer: 1024 ** 3 });
  if (r.error || r.status !== 0) {
    console.error(
      `git ${args.join(" ")} failed: ${r.error?.message ?? r.stderr}`,
    );
    process.exit(1);
  }
  return r.stdout ?? Buffer.alloc(0);
}

// Best-effort permission tightening: chmod is a no-op on Windows/ACL volumes.
function restrictPerms(target, perms) {
  try {
    chmodSync(target, perms);
  } catch {
    // no better fallback than the platform default
  }
}

// The git-ignored .env* files under `base`, across every ENV_DIRS entry. One
// batched `git check-ignore --stdin` rather than a fork per file. check-ignore
// consults the index, so tracked files (.env.default) are never reported —
// that is what keeps them out of both the symlink pass and the archive.
function listIgnoredEnvFiles(base) {
  const candidates = [];
  for (const dir of ENV_DIRS) {
    const dirPath = join(base, dir);
    if (!existsSync(dirPath)) continue;
    for (const name of readdirSync(dirPath)) {
      if (!name.startsWith(".env")) continue;
      candidates.push({
        dir,
        name,
        rel: dir ? `${dir}/${name}` : name,
        path: join(dirPath, name),
      });
    }
  }
  if (candidates.length === 0) return [];
  const r = git(
    ["check-ignore", "--stdin"],
    base,
    `${candidates.map((c) => c.rel).join("\n")}\n`,
  );
  // 0 = at least one path ignored, 1 = none ignored, anything else = git error
  if (r.status !== 0 && r.status !== 1) {
    console.error(
      `warning: could not determine which .env files are ignored in ${base} ` +
        `(${r.error?.message ?? r.stderr}); env files will not be linked, and ` +
        `local-only env edits will not be backed up`,
    );
    return [];
  }
  const ignored = new Set((r.stdout ?? "").split("\n").filter(Boolean));
  return candidates.filter((c) => ignored.has(c.rel));
}

function setup() {
  if (!root) {
    console.error(
      "!!! ORCA_ROOT_PATH not set: .env files were NOT linked and dependencies\n" +
        "!!! were NOT installed. This worktree is NOT ready to use. Re-run\n" +
        "!!! `ORCA_ROOT_PATH=<primary checkout> node .orca/hooks.mjs setup`.",
    );
    return;
  }

  // Symlink gitignored .env files from the primary checkout so edits propagate
  // to every worktree (tracked files like .env.default come with the checkout
  // and must not be linked — that would dirty git status with a typechange).
  let linked = 0;
  for (const env of listIgnoredEnvFiles(root)) {
    let srcStat;
    try {
      srcStat = statSync(env.path); // follows links: root's .env may itself be a symlink
    } catch {
      continue; // broken symlink
    }
    if (!srcStat.isFile()) continue;
    const destDir = join(worktree, env.dir);
    mkdirSync(destDir, { recursive: true });
    const dest = join(destDir, env.name);
    rmSync(dest, { force: true });
    try {
      symlinkSync(env.path, dest);
      linked++;
    } catch {
      copyFileSync(env.path, dest); // Windows without Developer Mode: no symlinks
      console.warn(
        `warning: could not symlink ${env.rel}, copied it instead. Edits to ` +
          `the copy do NOT reach the primary checkout and are overwritten the ` +
          `next time setup runs (archive() backs up a diverged copy first).`,
      );
    }
  }
  if (linked > 0) {
    console.log(
      `linked ${linked} .env file(s) from ${root} — these are shared, so ` +
        `editing env in this worktree changes it for every worktree`,
    );
  }

  for (const dir of [".vscode", ".auth", "autogpt_platform/frontend/.auth"]) {
    const src = join(root, dir);
    if (existsSync(src)) {
      cpSync(src, join(worktree, dir), { recursive: true });
    }
  }

  // .claude local config, excluding nested agent worktree checkouts
  const claudeSrc = join(root, ".claude");
  if (existsSync(claudeSrc)) {
    const skip = resolve(claudeSrc, "worktrees");
    cpSync(claudeSrc, join(worktree, ".claude"), {
      recursive: true,
      filter: (s) => resolve(s) !== skip,
    });
  }

  // Dependency install. generate:api is included so the frontend typecheck
  // pre-commit hook passes in a fresh worktree.
  if (!process.env.ORCA_HOOKS_SKIP_INSTALL) {
    const platform = join(worktree, "autogpt_platform");
    const backend = join(platform, "backend");
    const frontend = join(platform, "frontend");
    run("poetry install", join(platform, "autogpt_libs"));
    run("poetry install", backend);
    run("poetry run prisma generate", backend);
    run("pnpm install", frontend);
    run("pnpm generate:api", frontend);
  }
}

// Git-ignored .env files are invisible to git status/diff, so a worktree can
// look "clean" while carrying local env edits (e.g. the Windows copy fallback,
// or a deliberately divergent config). Collect any non-symlink ignored .env*
// whose content differs from the root checkout's copy so archive() backs it up.
function collectDivergentEnvFiles() {
  if (!root) return [];
  const out = [];
  for (const env of listIgnoredEnvFiles(worktree)) {
    let st;
    try {
      st = lstatSync(env.path);
    } catch {
      continue;
    }
    if (!st.isFile()) continue; // symlinks already live in the root checkout
    let same = false;
    try {
      // byte comparison, not utf8: a lossy decode must never make a diverged
      // file look identical to the root copy
      same = readFileSync(env.path).equals(readFileSync(join(root, env.rel)));
    } catch {
      // missing/unreadable in root -> treat as divergent
    }
    if (!same) out.push(env);
  }
  return out;
}

// Copy one file into the archive, preserving its exact bytes. Never throws:
// one unreadable file must not cost the developer the rest of the backup.
function copyInto(destRoot, rel, srcPath) {
  const dest = join(destRoot, rel);
  try {
    mkdirSync(dirname(dest), { recursive: true, mode: 0o700 });
    copyFileSync(srcPath, dest);
    restrictPerms(dest, 0o600);
    return true;
  } catch (err) {
    console.error(`could not back up ${rel}: ${err.message}`);
    return false;
  }
}

function restoreDoc({ outDir, branch, head, patches, untracked, envs }) {
  const steps = [
    ...patches.map((p) => `git apply --binary <archive>/${p}`),
    ...(untracked ? ["cp -R <archive>/untracked/. ."] : []),
    // Not an unconditional copy: these are local, git-ignored env files and
    // blindly restoring them can clobber a working config.
    ...(envs
      ? [
          "# review these first, then copy back the ones you want:",
          "#   cp -R <archive>/ignored-env/. .",
        ]
      : []),
  ];
  const apply = steps.length
    ? steps.join("\n")
    : "# nothing could be captured — check the archive hook output for errors";
  return `# Worktree archive: ${basename(outDir)}

Written by \`.orca/hooks.mjs archive\` just before Orca deleted the worktree
\`${worktree}\` (branch \`${branch}\`, at commit \`${head}\`).

> **This archive can contain secrets.** Diverged \`.env\` files and any untracked
> credential material are stored verbatim. It is created owner-only (0700 dirs,
> 0600 files) and is **never cleaned up automatically** — delete it yourself once
> you have recovered what you need.

## Contents

${patches.map((p) => `- \`${p}\` — tracked-file changes (\`git diff --binary\`).`).join("\n")}
${untracked ? `- \`untracked/\` — byte-for-byte copies of ${untracked} untracked file(s).\n` : ""}${envs ? `- \`ignored-env/\` — ${envs} git-ignored \`.env*\` file(s) whose contents had diverged from the primary checkout.\n` : ""}
## Restoring

From a checkout at \`${head}\`:

\`\`\`sh
${apply}
\`\`\`

Apply \`staged.patch\` before \`unstaged.patch\`: the first is HEAD→index, the
second is index→working tree. \`--binary\` is required — the patches embed the
contents of modified binary files.${
    envs
      ? "\n\nFiles under `ignored-env/` are not tracked by git; review them before copying\nthem back over the primary checkout's env."
      : ""
  }
`;
}

// Archive hooks are killed after 120s, so this only writes/copies bytes — no
// installs, no compression.
function archive() {
  const status = git(["status", "--porcelain"], worktree);
  if (status.error) {
    console.error(`git unavailable: ${status.error.message}`);
    process.exit(1);
  }
  if (status.status !== 0) {
    console.log("not a git worktree, nothing to back up");
    return;
  }
  const divergentEnvs = collectDivergentEnvFiles();
  if (!status.stdout.trim() && divergentEnvs.length === 0) {
    console.log("worktree clean, nothing to back up");
    return;
  }

  const name = (process.env.ORCA_WORKSPACE_NAME || basename(worktree)).replace(
    /[^\w.-]+/g,
    "_",
  );
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const backupsDir = join(
    dirname(root ?? worktree),
    "worktree-archive-backups",
  );
  const outDir = join(backupsDir, `${name}-${stamp}`);
  mkdirSync(outDir, { recursive: true });
  // The archive holds .env values and whatever secrets happened to be sitting
  // untracked in the worktree, so keep the whole tree owner-only.
  restrictPerms(backupsDir, 0o700);
  restrictPerms(outDir, 0o700);

  // --binary: without it git emits only "Binary files ... differ" and the new
  // bytes of a modified tracked binary are lost. Staged and unstaged changes go
  // to separate files so each one is genuinely applyable with `git apply`.
  const patches = [];
  for (const [file, args] of [
    ["staged.patch", ["diff", "--cached", "--binary"]],
    ["unstaged.patch", ["diff", "--binary"]],
  ]) {
    const body = gitStrictRaw(args, worktree);
    if (body.length === 0) continue; // git prints nothing for an empty diff
    const p = join(outDir, file);
    writeFileSync(p, body, { mode: 0o600 });
    restrictPerms(p, 0o600);
    patches.push(file);
  }

  // Untracked and diverged env files are copied byte-for-byte rather than
  // inlined as decoded text: a utf8 decode silently corrupts PNGs, sqlite DBs
  // and archives, and copyFileSync never buffers the file in the JS heap, so a
  // large untracked file cannot OOM the hook.
  const untracked = gitStrict(
    ["ls-files", "--others", "--exclude-standard"],
    worktree,
  )
    .split("\n")
    .filter(Boolean);
  let untrackedCount = 0;
  for (const rel of untracked) {
    if (copyInto(join(outDir, "untracked"), rel, join(worktree, rel))) {
      untrackedCount++;
    }
  }
  let envCount = 0;
  for (const e of divergentEnvs) {
    if (copyInto(join(outDir, "ignored-env"), e.rel, e.path)) envCount++;
  }

  const doc = join(outDir, "RESTORE.md");
  writeFileSync(
    doc,
    restoreDoc({
      outDir,
      branch: git(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        worktree,
      ).stdout?.trim(),
      head: git(["rev-parse", "HEAD"], worktree).stdout?.trim(),
      patches,
      untracked: untrackedCount,
      envs: envCount,
    }),
    { mode: 0o600 },
  );
  restrictPerms(doc, 0o600);
  console.log(
    `uncommitted work backed up to ${outDir} (${patches.length} patch file(s), ` +
      `${untrackedCount} untracked file(s), ${envCount} diverged env file(s)) — ` +
      `see RESTORE.md`,
  );
}

if (mode === "setup") {
  setup();
} else if (mode === "archive") {
  archive();
} else {
  console.error("usage: node .orca/hooks.mjs <setup|archive>");
  process.exit(2);
}
