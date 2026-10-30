import { expect, test } from "@playwright/test";

import { readProofConfig } from "./config";
import { checkRelease } from "./releases";

function environment(): NodeJS.ProcessEnv {
  return {
    AUTOGPT_E2E_REAL_FEED: "1", AUTOGPT_DESKTOP_UPDATES: "on",
    AUTOGPT_DESKTOP_VARIANT: "feed-proof", AUTOGPT_E2E_KIND: "nsis",
    AUTOGPT_E2E_INSTALLER: "AutoGPT-feed-proof-Setup-1.0.0-x64.exe",
    AUTOGPT_E2E_FEED_FROM_VERSION: "1.0.0", AUTOGPT_E2E_FEED_TO_VERSION: "1.0.1",
    AUTOGPT_E2E_FEED_FROM_SOURCE: "a".repeat(40), AUTOGPT_E2E_FEED_TO_SOURCE: "b".repeat(40),
    AUTOGPT_E2E_FEED_FROM_SHA256: "c".repeat(64), AUTOGPT_E2E_FEED_TO_SHA256: "d".repeat(64),
    AUTOGPT_E2E_FEED_TO_ARTIFACT: "AutoGPT-feed-proof-Setup-1.0.1-x64.exe",
  };
}

test("requires explicit real-feed opt-in with updates enabled", () => {
  for (const changed of [
    { AUTOGPT_E2E_REAL_FEED: undefined },
    { AUTOGPT_DESKTOP_UPDATES: "off" },
    { AUTOGPT_DESKTOP_VARIANT: "" },
    { AUTOGPT_E2E_KIND: "deb" },
    { AUTOGPT_E2E_FEED_TO_VERSION: "1.0.0" },
    { AUTOGPT_E2E_FEED_TO_VERSION: "1.0.1-rc.1" },
    { AUTOGPT_E2E_FEED_TO_SOURCE: "main" },
    { AUTOGPT_E2E_FEED_TO_SHA256: undefined },
    { AUTOGPT_E2E_FEED_TO_ARTIFACT: "../other.exe" },
  ]) expect(() => readProofConfig({ ...environment(), ...changed })).toThrow();
});

test("cannot point automatic restart at a developer runtime or temporary data override", () => {
  for (const key of ["AUTOGPT_DESKTOP_RUNTIME", "AUTOGPT_DESKTOP_DATA_DIR", "AUTOGPT_E2E_DATA_DIR", "AUTOGPT_E2E_INSTALL_DIR"]) {
    expect(() => readProofConfig({ ...environment(), [key]: "/some/path" })).toThrow(key);
  }
});

test("normal latest-feed proof needs an explicit disposable-machine choice", () => {
  const proof = readProofConfig({ ...environment(), AUTOGPT_DESKTOP_VARIANT: "", AUTOGPT_E2E_FEED_DISPOSABLE_NORMAL: "1" });
  expect(proof.tagPrefix).toBe("desktop-v");
  expect(readProofConfig(environment()).tagPrefix).toBe("desktop-feed-proof-v");
});

test("rejects draft, wrong source, wrong artifact and changed digest before installing A", () => {
  const source = "b".repeat(40);
  const hash = "d".repeat(64);
  const tag = "desktop-feed-proof-v1.0.1";
  const file = "AutoGPT-feed-proof-Setup-1.0.1-x64.exe";
  const release = {
    id: 1, tag_name: tag, draft: false, prerelease: true, body: `Built from \`${source}\``,
    assets: [{ name: file, size: 100, digest: `sha256:${hash}`, browser_download_url: `https://github.com/ntindle/autogpt/releases/download/${tag}/${file}` }],
  };
  expect(checkRelease(release, tag, source, file, hash).name).toBe(file);
  expect(() => checkRelease({ ...release, draft: true }, tag, source, file, hash)).toThrow();
  expect(() => checkRelease({ ...release, body: "wrong source" }, tag, source, file, hash)).toThrow();
  expect(() => checkRelease(release, tag, source, "other.exe", hash)).toThrow();
  expect(() => checkRelease(release, tag, source, file, "e".repeat(64))).toThrow();
});
