// Tests for the Orca worktree hooks. Self-contained: node:test + node:assert
// only, no repo test harness, no network, no package installs.
//
//   node --test .orca/hooks.test.mjs
//
// archive() is the data-loss path — it runs immediately before Orca deletes a
// worktree for good — so the assertions here are about faithfulness: every byte
// the developer had must come back out of the archive.
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HOOKS = join(dirname(fileURLToPath(import.meta.url)), "hooks.mjs");
const POSIX = process.platform !== "win32";
// Bytes that do not survive a utf8 round-trip: a lone 0xff, a NUL, a truncated
// UTF-8 sequence. Anything that inlines file contents as decoded text mangles
// these into U+FFFD and the "backup" becomes unrestorable.
const BINARY = Buffer.from([0x00, 0xff, 0xfe, 0x89, 0x50, 0x4e, 0x47, 0x0d]);

function git(args, cwd, opts = {}) {
  const r = spawnSync("git", args, { encoding: "utf8", cwd, ...opts });
  if (r.status !== 0) {
    throw new Error(`git ${args.join(" ")} failed: ${r.stderr ?? r.error}`);
  }
  return r.stdout ?? "";
}

function runHook(mode, env) {
  return spawnSync(process.execPath, [HOOKS, mode], {
    encoding: "utf8",
    env: { ...process.env, ORCA_HOOKS_SKIP_INSTALL: "1", ...env },
  });
}

/**
 * base/
 *   root/            primary checkout (committed .env.default, ignored .env)
 *   wt/              git worktree of root — the "child" worktree Orca manages
 *   worktree-archive-backups/   where archive() writes
 */
function makeFixture(t, { platformDirs = true } = {}) {
  const base = mkdtempSync(join(tmpdir(), "orca-hooks-"));
  t.after(() => rmSync(base, { recursive: true, force: true }));
  const root = join(base, "root");
  mkdirSync(root);
  git(["init", "-q", "-b", "main"], root);
  git(["config", "user.email", "hooks@test"], root);
  git(["config", "user.name", "hooks test"], root);
  git(["config", "commit.gpgsign", "false"], root);

  writeFileSync(join(root, ".gitignore"), ".env*\n!.env.default\n");
  writeFileSync(join(root, ".env.default"), "TRACKED=default\n");
  writeFileSync(join(root, ".env"), "ROOT_SECRET=from-root\n");
  writeFileSync(join(root, "tracked.txt"), "original\n");
  writeFileSync(join(root, "tracked.bin"), BINARY);
  if (platformDirs) {
    const backend = join(root, "autogpt_platform", "backend");
    mkdirSync(backend, { recursive: true });
    writeFileSync(join(backend, ".env"), "BACKEND_SECRET=from-root\n");
    writeFileSync(join(backend, "keep.txt"), "backend\n");
  }
  git(["add", "-A"], root);
  git(["commit", "-qm", "init"], root);

  const wt = join(base, "wt");
  git(["worktree", "add", "-q", "-b", "feature", wt], root);
  return {
    base,
    root,
    wt,
    env: { ORCA_ROOT_PATH: root, ORCA_WORKTREE_PATH: wt },
  };
}

function latestArchive(base) {
  const dir = join(base, "worktree-archive-backups");
  if (!existsSync(dir)) return null;
  const entries = readdirSync(dir).sort();
  return entries.length ? join(dir, entries[entries.length - 1]) : null;
}

test("setup symlinks ignored .env files and leaves tracked files alone", (t) => {
  const { root, wt, env } = makeFixture(t);
  const r = runHook("setup", env);
  assert.equal(r.status, 0, r.stderr);

  assert.ok(lstatSync(join(wt, ".env")).isSymbolicLink());
  assert.equal(
    readFileSync(join(wt, ".env"), "utf8"),
    "ROOT_SECRET=from-root\n",
  );
  assert.ok(
    lstatSync(join(wt, "autogpt_platform/backend/.env")).isSymbolicLink(),
    "nested ENV_DIRS entries are linked too",
  );

  // .env.default is tracked, so check-ignore must not report it: linking it
  // would show up as a typechange in git status.
  assert.ok(!lstatSync(join(root, ".env.default")).isSymbolicLink());
  assert.ok(!lstatSync(join(wt, ".env.default")).isSymbolicLink());
  assert.equal(git(["status", "--porcelain"], wt), "");
});

test("setup copies .claude but skips the nested worktrees checkout", (t) => {
  const { root, wt, env } = makeFixture(t);
  mkdirSync(join(root, ".claude", "worktrees", "nested"), { recursive: true });
  writeFileSync(join(root, ".claude", "settings.json"), "{}\n");
  writeFileSync(join(root, ".claude", "worktrees", "nested", "huge"), "x");

  assert.equal(runHook("setup", env).status, 0);
  assert.ok(existsSync(join(wt, ".claude", "settings.json")));
  assert.ok(!existsSync(join(wt, ".claude", "worktrees")));
});

test("setup skips install steps whose directory does not exist", (t) => {
  // No autogpt_platform at all, and installs NOT skipped: every run() must
  // no-op instead of exiting non-zero on spawn ENOENT.
  const { env } = makeFixture(t, { platformDirs: false });
  const r = runHook("setup", { ...env, ORCA_HOOKS_SKIP_INSTALL: "" });
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /skipping \(no such directory/);
  assert.doesNotMatch(r.stdout, /\$ poetry install/);
});

test("setup fails loudly when ORCA_ROOT_PATH is missing", (t) => {
  const { wt } = makeFixture(t);
  const r = runHook("setup", {
    ORCA_WORKTREE_PATH: wt,
    ORCA_ROOT_PATH: undefined,
  });
  assert.match(r.stderr, /ORCA_ROOT_PATH not set/);
  assert.match(r.stderr, /NOT ready to use/);
});

test("archive writes nothing for a clean worktree", (t) => {
  const { base, wt, env } = makeFixture(t);
  runHook("setup", env);
  const r = runHook("archive", env);
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /worktree clean/);
  assert.equal(latestArchive(base), null);
  assert.equal(git(["status", "--porcelain"], wt), "");
});

test("archive round-trips staged, unstaged and binary tracked changes", (t) => {
  const { base, root, wt, env } = makeFixture(t);
  runHook("setup", env);

  // staged change
  writeFileSync(join(wt, "tracked.txt"), "staged edit\n");
  git(["add", "tracked.txt"], wt);
  // unstaged change on top of it
  writeFileSync(join(wt, "tracked.txt"), "staged edit\nthen unstaged\n");
  // modified tracked binary — plain `git diff` would only say "Binary files differ"
  const newBinary = Buffer.from([0xff, 0x00, 0x01, 0x02, 0x80, 0xfe]);
  writeFileSync(join(wt, "tracked.bin"), newBinary);

  const r = runHook("archive", env);
  assert.equal(r.status, 0, r.stderr);
  const out = latestArchive(base);
  assert.ok(out, "archive directory created");
  assert.ok(existsSync(join(out, "staged.patch")));
  assert.ok(existsSync(join(out, "unstaged.patch")));
  assert.match(
    readFileSync(join(out, "unstaged.patch"), "utf8"),
    /GIT binary patch/,
    "binary contents are embedded, not just a 'differ' marker",
  );

  // Replay the documented restore procedure into a pristine checkout.
  const replay = join(base, "replay");
  git(["worktree", "add", "-q", "--detach", replay, "feature"], root);
  git(["apply", "--binary", join(out, "staged.patch")], replay);
  git(["apply", "--binary", join(out, "unstaged.patch")], replay);
  assert.equal(
    readFileSync(join(replay, "tracked.txt"), "utf8"),
    "staged edit\nthen unstaged\n",
  );
  assert.ok(
    readFileSync(join(replay, "tracked.bin")).equals(newBinary),
    "tracked binary bytes survive the archive/restore round trip",
  );
});

test("archive round-trips a tracked text file that is not valid UTF-8", (t) => {
  const { base, root, wt, env } = makeFixture(t);
  runHook("setup", env);

  // git only base85-encodes blobs it classifies as *binary*, and it classifies
  // by looking for a NUL byte. LATIN1 has none, so git calls it text and emits
  // its bytes into `git diff --binary` verbatim — no base85 armour. If the hook
  // decodes that patch as utf8, 0xe9/0xef become U+FFFD, the patch no longer
  // matches the blob it came from, and `git apply` rejects it: the developer's
  // only copy of this file's changes is gone.
  const LATIN1 = Buffer.from("caf\xe9 na\xefve\n", "latin1");
  const LATIN1_EDIT = Buffer.from("caf\xe9 na\xefve edited\n", "latin1");
  writeFileSync(join(wt, "latin1.txt"), LATIN1);
  git(["add", "latin1.txt"], wt);
  git(["commit", "-qm", "add latin1"], wt);
  writeFileSync(join(wt, "latin1.txt"), LATIN1_EDIT);

  const r = runHook("archive", env);
  assert.equal(r.status, 0, r.stderr);
  const out = latestArchive(base);
  assert.ok(existsSync(join(out, "unstaged.patch")));

  const replay = join(base, "replay");
  git(["worktree", "add", "-q", "--detach", replay, "feature"], root);
  git(["apply", "--binary", join(out, "unstaged.patch")], replay);
  assert.ok(
    readFileSync(join(replay, "latin1.txt")).equals(LATIN1_EDIT),
    "non-UTF-8 tracked text must survive the archive/restore round trip",
  );
});

test("archive preserves untracked binary files byte-for-byte", (t) => {
  const { base, wt, env } = makeFixture(t);
  runHook("setup", env);
  mkdirSync(join(wt, "nested", "deep"), { recursive: true });
  writeFileSync(join(wt, "nested", "deep", "image.png"), BINARY);
  writeFileSync(join(wt, "notes.txt"), "scratch\n");

  assert.equal(runHook("archive", env).status, 0);
  const out = latestArchive(base);
  assert.ok(
    readFileSync(join(out, "untracked", "nested", "deep", "image.png")).equals(
      BINARY,
    ),
    "untracked binary must not be utf8-decoded into the backup",
  );
  assert.equal(
    readFileSync(join(out, "untracked", "notes.txt"), "utf8"),
    "scratch\n",
  );
});

test(
  "archive keeps going when an untracked file cannot be read",
  { skip: !POSIX },
  (t) => {
    const { base, wt, env } = makeFixture(t);
    runHook("setup", env);
    symlinkSync(join(wt, "does-not-exist"), join(wt, "dangling"));
    writeFileSync(join(wt, "survivor.txt"), "keep me\n");

    const r = runHook("archive", env);
    assert.equal(r.status, 0, r.stderr);
    assert.match(r.stderr, /could not back up dangling/);
    const out = latestArchive(base);
    assert.equal(
      readFileSync(join(out, "untracked", "survivor.txt"), "utf8"),
      "keep me\n",
      "one unreadable file must not cost the rest of the backup",
    );
  },
);

test("archive backs up a git-clean worktree carrying a diverged ignored .env", (t) => {
  const { base, wt, env } = makeFixture(t);
  runHook("setup", env);

  // Exactly the state the Windows copy fallback produces: a real file, not a
  // symlink, ignored by git, with content the root checkout does not have.
  rmSync(join(wt, ".env"));
  writeFileSync(join(wt, ".env"), "ROOT_SECRET=edited-locally\n");
  assert.equal(
    git(["status", "--porcelain"], wt),
    "",
    "git still sees it clean",
  );

  const r = runHook("archive", env);
  assert.equal(r.status, 0, r.stderr);
  const out = latestArchive(base);
  assert.ok(out, "a clean-but-diverged worktree still produces an archive");
  assert.equal(
    readFileSync(join(out, "ignored-env", ".env"), "utf8"),
    "ROOT_SECRET=edited-locally\n",
  );
  // An identical (non-diverged) copy is not worth archiving.
  assert.ok(!existsSync(join(out, "ignored-env", "autogpt_platform")));
  // This archive has no patches and no untracked files, so the restore block
  // would be empty if ignored-env were left out of it.
  assert.match(
    readFileSync(join(out, "RESTORE.md"), "utf8"),
    /cp -R <archive>\/ignored-env\/\. \./,
  );
});

test(
  "archive restricts the backup tree to the owner",
  { skip: !POSIX },
  (t) => {
    const { base, wt, env } = makeFixture(t);
    runHook("setup", env);
    writeFileSync(join(wt, "tracked.txt"), "TOKEN=hunter2\n");
    writeFileSync(join(wt, "untracked.txt"), "SECRET=hunter2\n");
    rmSync(join(wt, ".env"));
    writeFileSync(join(wt, ".env"), "ROOT_SECRET=diverged\n");

    assert.equal(runHook("archive", env).status, 0);
    const out = latestArchive(base);
    const perms = (p) => statSync(p).mode & 0o777;
    assert.equal(perms(join(base, "worktree-archive-backups")), 0o700);
    assert.equal(perms(out), 0o700);
    assert.equal(perms(join(out, "unstaged.patch")), 0o600);
    assert.equal(perms(join(out, "untracked", "untracked.txt")), 0o600);
    assert.equal(perms(join(out, "ignored-env", ".env")), 0o600);
    assert.equal(perms(join(out, "RESTORE.md")), 0o600);
  },
);

test("archive documents how to restore what it wrote", (t) => {
  const { base, wt, env } = makeFixture(t);
  runHook("setup", env);
  writeFileSync(join(wt, "tracked.txt"), "changed\n");
  writeFileSync(join(wt, "extra.txt"), "new\n");

  assert.equal(runHook("archive", env).status, 0);
  const doc = readFileSync(join(latestArchive(base), "RESTORE.md"), "utf8");
  assert.match(doc, /git apply --binary <archive>\/unstaged\.patch/);
  assert.match(doc, /cp -R <archive>\/untracked\/\. \./);
  assert.match(doc, /can contain secrets/);
  assert.match(doc, new RegExp(git(["rev-parse", "HEAD"], wt).trim()));
});

test("archive is a no-op outside a git worktree", (t) => {
  const { base } = makeFixture(t);
  const plain = join(base, "plain");
  mkdirSync(plain);
  const r = runHook("archive", {
    ORCA_WORKTREE_PATH: plain,
    ORCA_ROOT_PATH: undefined,
  });
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /not a git worktree/);
});

// chmod 000 is not honoured for root, so this only proves the degrade path for
// an ordinary user.
test(
  "archive degrades gracefully on an unreadable diverged env",
  { skip: !POSIX || process.getuid?.() === 0 },
  (t) => {
    const { base, wt, env } = makeFixture(t);
    runHook("setup", env);
    rmSync(join(wt, ".env"));
    writeFileSync(join(wt, ".env"), "ROOT_SECRET=diverged\n");
    writeFileSync(join(wt, "untracked.txt"), "keep me\n");
    chmodSync(join(wt, ".env"), 0o000);

    const r = runHook("archive", env);
    assert.equal(r.status, 0, r.stderr);
    const out = latestArchive(base);
    assert.ok(out, "the rest of the backup is still written");
    assert.equal(
      readFileSync(join(out, "untracked", "untracked.txt"), "utf8"),
      "keep me\n",
    );
  },
);
