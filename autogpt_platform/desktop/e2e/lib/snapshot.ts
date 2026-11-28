// A record of an installed tree, to prove that running the app changes
// nothing in it: system-wide installs are read-only, and on macOS a changed
// file breaks the code signature. The same check build/smoke_test.py makes on
// the unpackaged runtime (`snapshot` and `changed_files` there).

import fs from "node:fs";
import path from "node:path";

/** Relative path -> "size:mtime in ns" for files, "dir" for directories. */
export type Snapshot = Map<string, string>;

export function takeSnapshot(root: string): Snapshot {
  const entries: Snapshot = new Map();
  walk(root, "", entries);
  return entries;
}

export function changedPaths(before: Snapshot, after: Snapshot): string[] {
  const names = new Set([...before.keys(), ...after.keys()]);
  return [...names].filter((name) => before.get(name) !== after.get(name)).sort();
}

export function summarize(changed: string[]): string {
  const listed = changed.slice(0, 20).join("\n  ");
  return `${changed.length} path(s) changed:\n  ${listed}${changed.length > 20 ? "\n  ..." : ""}`;
}

function walk(root: string, relative: string, entries: Snapshot): void {
  for (const entry of fs.readdirSync(path.join(root, relative), { withFileTypes: true })) {
    const name = relative ? path.join(relative, entry.name) : entry.name;
    if (entry.isDirectory()) {
      entries.set(name, "dir");
      walk(root, name, entries);
    } else if (entry.isFile()) {
      const status = fs.lstatSync(path.join(root, name), { bigint: true });
      entries.set(name, `${status.size}:${status.mtimeNs}`);
    } else {
      entries.set(name, "other");
    }
  }
}
