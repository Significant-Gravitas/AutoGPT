import { describe, expect, it } from "vitest";
import { hasAllRequiredScopes } from "../useGoogleDrivePicker";

describe("hasAllRequiredScopes", () => {
  it("returns true when no scopes are required", () => {
    expect(hasAllRequiredScopes([], undefined)).toBe(true);
    expect(hasAllRequiredScopes(["drive.file"], undefined)).toBe(true);
    expect(hasAllRequiredScopes([], [])).toBe(true);
  });

  it("returns true when granted scopes are a superset of required", () => {
    expect(
      hasAllRequiredScopes(
        ["drive.file", "drive.metadata", "drive.readonly"],
        ["drive.file", "drive.metadata"],
      ),
    ).toBe(true);
  });

  it("returns true when granted scopes exactly match required", () => {
    expect(
      hasAllRequiredScopes(
        ["drive.file", "drive.metadata"],
        ["drive.file", "drive.metadata"],
      ),
    ).toBe(true);
  });

  it("returns false when even one required scope is missing", () => {
    // Regression: the picker must gate on EVERY required scope being
    // present. Prior to the fix the check used
    // `Set.prototype.isSupersetOf` — which is ES2025 and not yet in
    // the supported browser baseline — so the check silently ran as
    // `undefined` and never failed closed.
    expect(
      hasAllRequiredScopes(["drive.file"], ["drive.file", "drive.metadata"]),
    ).toBe(false);
  });

  it("returns false when granted scopes are empty but required is non-empty", () => {
    expect(hasAllRequiredScopes([], ["drive.file"])).toBe(false);
  });

  it("tolerates null / undefined granted scopes", () => {
    // Shape from the API: `oauth2` credentials sometimes store `scopes`
    // as null.  We should treat null/undefined as "no scopes granted",
    // not crash.
    expect(hasAllRequiredScopes(null, ["drive.file"])).toBe(false);
    expect(hasAllRequiredScopes(undefined, ["drive.file"])).toBe(false);
    expect(hasAllRequiredScopes(null, [])).toBe(true);
    expect(hasAllRequiredScopes(null, undefined)).toBe(true);
  });
});
