import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { SDK_KEY_SUFFIX, UNKNOWN_KEY_PREFIX } from "../helpers";

// Rerun reads two of the backend's conventions. These fail when the backend
// changes one, instead of Rerun quietly no longer recognising it.
function backendSource(path: string) {
  return readFileSync(resolve("../backend/backend", path), "utf8");
}

describe("the backend conventions Rerun reads", () => {
  it("a refusal names an unknown key with the prefix Rerun parses", () => {
    const source = backendSource("executor/utils.py");
    expect(source).toContain(
      `CRED_ERR_UNKNOWN_PREFIX = ${JSON.stringify(UNKNOWN_KEY_PREFIX)}`,
    );
    expect(source).toContain(
      'f"{CRED_ERR_UNKNOWN_PREFIX}{credentials_meta.id}"',
    );
  });

  it("a provider's own platform key ends with the suffix Rerun recognises", () => {
    expect(backendSource("data/model.py")).toContain(
      `SDK_DEFAULT_SUFFIX = ${JSON.stringify(SDK_KEY_SUFFIX)}`,
    );
  });
});
