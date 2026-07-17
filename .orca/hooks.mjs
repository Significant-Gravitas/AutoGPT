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
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const mode = process.argv[2];
const worktree = process.env.ORCA_WORKTREE_PATH ?? process.cwd();
const root = process.env.ORCA_ROOT_PATH;

function run(cmd, cwd) {
  console.log(`$ ${cmd}  (in ${cwd})`);
  const r = spawnSync(cmd, { shell: true, stdio: "inherit", cwd });
  if (r.status !== 0) {
    console.error(`command failed with status ${r.status}: ${cmd}`);
    process.exit(r.status ?? 1);
  }
}

function git(args, cwd) {
  return spawnSync("git", args, { encoding: "utf8", cwd });
}

function setup() {
  if (!root) {
    console.error("ORCA_ROOT_PATH not set; skipping setup");
    return;
  }

  // Symlink gitignored .env files from the primary checkout so edits propagate
  // to every worktree (tracked files like .env.default come with the checkout
  // and must not be linked — that would dirty git status with a typechange).
  const envDirs = [
    "",
    "autogpt_platform",
    "autogpt_platform/backend",
    "autogpt_platform/frontend",
    "autogpt_platform/db/docker",
  ];
  for (const dir of envDirs) {
    const srcDir = join(root, dir);
    if (!existsSync(srcDir)) continue;
    for (const name of readdirSync(srcDir)) {
      if (!name.startsWith(".env")) continue;
      const src = join(srcDir, name);
      if (!lstatSync(src).isFile()) continue;
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

// Archive hooks are killed after 120s, so this only dumps text — no installs.
function archive() {
  const status = git(["status", "--porcelain"], worktree);
  if (status.status !== 0) {
    console.log("not a git worktree, nothing to back up");
    return;
  }
  if (!status.stdout.trim()) {
    console.log("worktree clean, nothing to back up");
    return;
  }

  const outDir = join(dirname(root ?? worktree), "worktree-archive-backups");
  mkdirSync(outDir, { recursive: true });
  const name = process.env.ORCA_WORKSPACE_NAME || basename(worktree);
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const outPath = join(outDir, `${name.replace(/[^\w.-]+/g, "_")}-${stamp}.patch`);

  const parts = [
    git(["diff"], worktree).stdout ?? "",
    git(["diff", "--cached"], worktree).stdout ?? "",
  ];
  const untracked = (git(["ls-files", "--others", "--exclude-standard"], worktree)
    .stdout ?? "")
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
