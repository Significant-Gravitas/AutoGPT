import { describe, expect, test } from "vitest";

import { maskAPIKey } from "../helpers";

describe("maskAPIKey", () => {
  test("joins head, a fixed mask, and tail", () => {
    expect(maskAPIKey("pk_live_", "abcd1234")).toBe("pk_live_••••••••abcd1234");
  });

  test("handles empty head and tail", () => {
    expect(maskAPIKey("", "")).toBe("••••••••");
  });
});
