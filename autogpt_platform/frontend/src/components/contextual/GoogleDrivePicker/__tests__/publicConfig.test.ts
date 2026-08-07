import { afterEach, describe, expect, it, vi } from "vitest";

import { fetchGooglePickerPublicConfig } from "../publicConfig";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchGooglePickerPublicConfig", () => {
  it.each([
    ["an array", []],
    ["a non-string field", { clientId: 42 }],
  ])("rejects %s response", async (_description, payload) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(Response.json(payload)));

    await expect(fetchGooglePickerPublicConfig()).rejects.toThrow(
      "Failed to load Google Picker runtime configuration.",
    );
  });
});
