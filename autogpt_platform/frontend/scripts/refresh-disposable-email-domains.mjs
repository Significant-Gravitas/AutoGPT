// Refreshes src/lib/auth/disposable-email-domains.json from
// github.com/disposable-email-domains/disposable-email-domains (CC0 1.0).
// Run: node scripts/refresh-disposable-email-domains.mjs, then commit the diff.
import { writeFile } from "node:fs/promises";

const REPO = "disposable-email-domains/disposable-email-domains";
const OUTPUT = new URL(
  "../src/lib/auth/disposable-email-domains.json",
  import.meta.url,
);

const commit = await fetchJSON(
  `https://api.github.com/repos/${REPO}/commits/main`,
);
const text = await fetchText(
  `https://raw.githubusercontent.com/${REPO}/${commit.sha}/disposable_email_blocklist.conf`,
);
const domains = [
  ...new Set(
    text
      .split("\n")
      .map((line) => line.trim().toLowerCase())
      .filter((line) => line && !line.startsWith("#")),
  ),
].sort();

await writeFile(
  OUTPUT,
  `${JSON.stringify({ source: `https://github.com/${REPO}`, updated: commit.commit.committer.date, domains }, null, 2)}\n`,
);
console.log(`${domains.length} domains from ${commit.sha}`);

async function fetchJSON(url) {
  return JSON.parse(await fetchText(url));
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.text();
}
