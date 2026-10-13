import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../../..");
if (!process.argv[2])
  throw new Error("Pass the absolute capture-manifest.json path.");
const manifestPath = path.resolve(process.argv[2]);
const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
const digest = async (file) =>
  crypto
    .createHash("sha256")
    .update(await fs.readFile(file))
    .digest("hex");
const stale = [];
for (const { path: file, sha256: hash } of manifest.componentSource
  .sourceFiles) {
  if ((await digest(path.join(repo, file))) !== hash) stale.push(file);
}
for (const capture of manifest.captures) {
  if (
    (await digest(path.join(path.dirname(manifestPath), capture.file))) !==
    capture.sha256
  )
    stale.push(capture.file);
}
if (stale.length)
  throw new Error(
    `Regenerate screenshots: source or PNG changed:\n${stale.join("\n")}`,
  );
if (
  manifest.errors.length ||
  !manifest.cleanup.browserClosed ||
  !manifest.cleanup.serverClosed
)
  throw new Error("Capture did not complete cleanly.");
console.log(
  JSON.stringify({
    verifiedSources: manifest.componentSource.sourceFiles.length,
    verifiedScreenshots: manifest.captures.length,
  }),
);
