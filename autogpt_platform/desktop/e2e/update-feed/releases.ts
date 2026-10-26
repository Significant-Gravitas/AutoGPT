import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

import { newestRelease } from "../../src/updater";
import { identity, isOwnUpdateFile } from "../../src/identity";
import type { ProofConfig } from "./config";

const API = "https://api.github.com/repos/ntindle/autogpt";
const DOWNLOADS = "https://github.com/ntindle/autogpt/releases/download/";

interface Release {
  id: number;
  tag_name: string;
  draft: boolean;
  prerelease: boolean;
  body: string;
  assets: { name: string; size: number; browser_download_url: string; digest: string | null }[];
}

async function response(url: string) {
  const result = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!result.ok) throw new Error(`${url}: HTTP ${result.status}`);
  return result;
}

export function checkRelease(release: Release, tag: string, source: string, artifact: string, sha256: string) {
  if (release.draft || release.tag_name !== tag) throw new Error(`Expected published release ${tag}`);
  if (!release.body.includes(`\`${source}\``)) throw new Error(`${tag} does not identify the pinned product source ${source}`);
  const asset = release.assets.find((item) => item.name === artifact);
  if (!asset || asset.size <= 0 || asset.browser_download_url !== `${DOWNLOADS}${tag}/${artifact}`) {
    throw new Error(`${tag} lacks the expected public artifact ${artifact}`);
  }
  if (asset.digest && asset.digest !== `sha256:${sha256}`) throw new Error(`${artifact}: GitHub digest differs from the pinned SHA-256`);
  return asset;
}

export async function checkFeed(config: ProofConfig, fromArtifact: string) {
  const product = identity(config.variant);
  if (!isOwnUpdateFile(product, config.fromVersion, fromArtifact) || !isOwnUpdateFile(product, config.toVersion, config.toArtifact)) {
    throw new Error("Pinned artifacts must belong to this installation and their expected versions");
  }
  const [a, b] = await Promise.all([config.fromVersion, config.toVersion].map(async (version) =>
    (await (await response(`${API}/releases/tags/${config.tagPrefix}${version}`)).json()) as Release,
  ));
  checkRelease(a, `${config.tagPrefix}${config.fromVersion}`, config.fromSource, fromArtifact, config.fromSha256);
  checkRelease(b, `${config.tagPrefix}${config.toVersion}`, config.toSource, config.toArtifact, config.toSha256);
  const sums = await (await response(`${DOWNLOADS}${b.tag_name}/SHA256SUMS`)).text();
  if (!sums.split(/\r?\n/).includes(`${config.toSha256}  ${config.toArtifact}`)) {
    throw new Error("Published SHA256SUMS does not match the independently pinned B artifact");
  }
  if (!config.variant) {
    const latest = await (await response(`${API}/releases/latest`)).json() as Release;
    if (latest.id !== b.id || b.prerelease) throw new Error("B must be the public normal app's latest release");
  } else {
    if (!b.prerelease) throw new Error("Variant B must remain a prerelease, isolated from the normal feed");
    const releases: Release[] = [];
    for (let page = 1; page <= 5; page += 1) {
      const listed = await (await response(`${API}/releases?per_page=100&page=${page}`)).json() as Release[];
      releases.push(...listed);
      if (listed.length < 100) break;
    }
    if (newestRelease(releases, identity(config.variant))?.tag !== b.tag_name) {
      throw new Error("B is not the variant's highest published stable version");
    }
  }
  return { a: { id: a.id, tag: a.tag_name }, b: { id: b.id, tag: b.tag_name } };
}

export async function sha256(file: string) {
  const hash = createHash("sha256");
  for await (const bytes of createReadStream(file)) hash.update(bytes);
  return hash.digest("hex");
}
