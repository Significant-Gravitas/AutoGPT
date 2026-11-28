// @vitest-environment node
import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";

// From this file, wherever the tests run from.
const GOOGLE_BLOCKS = fileURLToPath(
  new URL(
    "../../../../../../../backend/backend/blocks/google",
    import.meta.url,
  ),
);
const GOOGLE_SCOPE = /https:\/\/www\.googleapis\.com\/auth\/[a-z._]+/g;

/**
 * The scopes the Google blocks require, read from their
 * `GoogleCredentialsField([...])` declarations. A scope only reaches the
 * blocks on dev once Google has approved the app for it, so these are the
 * ones Google's consent screen shows without a warning.
 */
function scopesTheBlocksRequire() {
  const required = new Set<string>();
  for (const name of readdirSync(GOOGLE_BLOCKS)) {
    if (!name.endsWith(".py") || name.endsWith("_test.py")) continue;
    const source = readFileSync(resolve(GOOGLE_BLOCKS, name), "utf8");
    for (const field of source.matchAll(
      /GoogleCredentialsField\(\s*\[([^\]]*)\]/g,
    )) {
      for (const scope of field[1].match(GOOGLE_SCOPE) ?? []) {
        required.add(scope);
      }
    }
  }
  return required;
}

describe("the scopes onboarding asks Google for", () => {
  it("are all ones a Google block already requires", () => {
    const required = scopesTheBlocksRequire();
    // The reading itself still works: Gmail, Drive and Calendar declare some.
    expect(required.size).toBeGreaterThan(0);
    const unapproved = GOOGLE_ACCESS_SCOPES.filter(
      (scope) => !required.has(scope),
    );
    expect(unapproved).toEqual([]);
  });

  it("leave out Gmail's metadata scope, which limits every Gmail read", () => {
    expect(GOOGLE_ACCESS_SCOPES).not.toContain(
      "https://www.googleapis.com/auth/gmail.metadata",
    );
  });
});
