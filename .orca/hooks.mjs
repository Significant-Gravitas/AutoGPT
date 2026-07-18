#!/usr/bin/env node
// Orca worktree hooks for AutoGPT, invoked from orca.yaml as
// `node .orca/hooks.mjs <setup|archive>`. Node is the entry point (rather than
// a shell script) because Orca runs hooks under /bin/bash on macOS/Linux but
// cmd.exe on Windows, and `node` resolves identically in both.
import {
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

const ENV_DIRS = [
  "",
  "autogpt_platform",
  "autogpt_platform/backend",
  "autogpt_platform/frontend",
  "autogpt_platform/db/docker",
];

function run(cmd, cwd) {
  console.log(`$ ${cmd}  (in ${cwd})`);
  const r = spawnSync(cmd, { shell: true, stdio: "inherit", cwd });
  if (r.status !== 0) {
    console.error(`command failed with status ${r.status}: ${cmd}`);
    process.exit(r.status ?? 1);
  }
}

function git(args, cwd) {
  // maxBuffer: worktrees can carry very large uncommitted diffs; the 1MB
  // spawnSync default would silently truncate exactly what archive() backs up
  return spawnSync("git", args, { encoding: "utf8", cwd, maxBuffer: 1024 ** 3 });
}

function gitStrict(args, cwd) {
  const r = git(args, cwd);
  if (r.error || r.status !== 0) {
    console.error(`git ${args.join(" ")} failed: ${r.error?.message ?? r.stderr}`);
    process.exit(1);
  }
  return r.stdout ?? "";
}

function setup() {
  if (!root) {
    console.error("ORCA_ROOT_PATH not set; skipping setup");
    return;
  }

  // Symlink gitignored .env files from the primary checkout so edits propagate
  // to every worktree (tracked files like .env.default come with the checkout
  // and must not be linked — that would dirty git status with a typechange).
  for (const dir of ENV_DIRS) {
    const srcDir = join(root, dir);
    if (!existsSync(srcDir)) continue;
    for (const name of readdirSync(srcDir)) {
      if (!name.startsWith(".env")) continue;
      const src = join(srcDir, name);
      let srcStat;
      try {
        srcStat = statSync(src); // follows links: root's .env may itself be a symlink
      } catch {
        continue; // broken symlink
      }
      if (!srcStat.isFile()) continue;
      if (git(["-C", root, "check-ignore", "-q", src]).status !== 0) continue;
      const destDir = join(worktree, dir);
      mkdirSync(destDir, { recursive: true });
      const dest = join(destDir, name);
      rmSync(dest, { force: true });
      try {
        symlinkSync(src, dest);
      } catch {
        copyFileSync(src, dest); // Windows without Developer Mode: no symlinks
      }
    }
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

  // Dependency install (was branchlet postCreateCmd, plus generate:api so the
  // frontend typecheck pre-commit hook works in fresh worktrees).
  if (!process.env.ORCA_HOOKS_SKIP_INSTALL) {
    run("poetry install", join(worktree, "autogpt_platform/autogpt_libs"));
    run("poetry install", join(worktree, "autogpt_platform/backend"));
    run("poetry run prisma generate", join(worktree, "autogpt_platform/backend"));
    run("pnpm install", join(worktree, "autogpt_platform/frontend"));
    run("pnpm generate:api", join(worktree, "autogpt_platform/frontend"));
  }
}

// Git-ignored .env files are invisible to git status/diff, so a worktree can
// look "clean" while carrying local env edits (e.g. the Windows copy fallback,
// or a deliberately divergent config). Collect any non-symlink ignored .env*
// whose content differs from the root checkout's copy so archive() backs it up.
function collectDivergentEnvFiles() {
  if (!root) return [];
  const out = [];
  for (const dir of ENV_DIRS) {
    const wtDir = join(worktree, dir);
    if (!existsSync(wtDir)) continue;
    for (const name of readdirSync(wtDir)) {
      if (!name.startsWith(".env")) continue;
      const f = join(wtDir, name);
      let st;
      try {
        st = lstatSync(f);
      } catch {
        continue;
      }
      if (!st.isFile()) continue; // symlinks already live in the root checkout
      const rel = dir ? `${dir}/${name}` : name;
      if (git(["check-ignore", "-q", rel], worktree).status !== 0) continue; // tracked files are covered by git diff
      let same = false;
      try {
        same = readFileSync(f, "utf8") === readFileSync(join(root, dir, name), "utf8");
      } catch {
        // missing/unreadable in root -> treat as divergent
      }
      if (!same) out.push({ rel, path: f });
    }
  }
  return out;
}

// Archive hooks are killed after 120s, so this only dumps text — no installs.
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

  const outDir = join(dirname(root ?? worktree), "worktree-archive-backups");
  mkdirSync(outDir, { recursive: true });
  const name = process.env.ORCA_WORKSPACE_NAME || basename(worktree);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = join(outDir, `${name.replace(/[^\w.-]+/g, "_")}-${stamp}.patch`);

  const parts = [
    gitStrict(["diff"], worktree),
    gitStrict(["diff", "--cached"], worktree),
  ];
  const untracked = gitStrict(
    ["ls-files", "--others", "--exclude-standard"],
    worktree,
  )
    .split("\n")
    .filter(Boolean);
  for (const f of untracked) {
    let content;
    try {
      content = readFileSync(join(worktree, f), "utf8");
    } catch (err) {
      content = `<< unreadable: ${err.message} >>\n`;
    }
    parts.push(`=== UNTRACKED: ${f} ===\n${content}`);
  }
  for (const e of divergentEnvs) {
    parts.push(
      `=== IGNORED-LOCAL: ${e.rel} (differs from root checkout) ===\n${readFileSync(e.path, "utf8")}`,
    );
  }

  writeFileSync(outPath, parts.filter(Boolean).join("\n"));
  console.log(`uncommitted work backed up to ${outPath}`);
}

if (mode === "setup") {
  setup();
} else if (mode === "archive") {
  archive();
} else {
  console.error("usage: node .orca/hooks.mjs <setup|archive>");
  process.exit(2);
}
