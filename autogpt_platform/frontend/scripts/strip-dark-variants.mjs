#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const SRC = path.resolve(import.meta.dirname, "../src");
const SKIP = new Set(["node_modules", ".next", "__generated__", "coverage", "dist"]);

function walk(dir, files) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (SKIP.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, files);
    else if (/\.(tsx|ts|css)$/.test(entry.name)) files.push(full);
  }
}

function stripDark(source) {
  return source.replace(
    /(^|[\s"'`])([^\s"'`]*dark:[^\s"'`]*)/g,
    (match, lead, token) => {
      if (!/(?:^|:)dark:/.test(token)) return match;
      return lead === " " || lead === "\t" ? "" : lead;
    },
  );
}

const files = [];
walk(SRC, files);
let changed = 0;
let removed = 0;
for (const file of files) {
  const source = fs.readFileSync(file, "utf8");
  const next = stripDark(source);
  if (next === source) continue;
  removed += source.split("dark:").length - next.split("dark:").length;
  fs.writeFileSync(file, next);
  changed += 1;
}
console.log(JSON.stringify({ files: files.length, changed, removed }));
