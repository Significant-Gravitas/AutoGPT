import { describe, expect, it } from "vitest";
import { shouldBypassImageOptimization } from "./image";

describe("shouldBypassImageOptimization", () => {
  it.each(["/api/store/media/image.jpeg", "/_agpt/api/store/media/image.jpeg"])(
    "bypasses optimization for same-origin path %s",
    (src) => {
      expect(shouldBypassImageOptimization(src)).toBe(true);
    },
  );

  it.each([
    "https://storage.googleapis.com/bucket/image.jpeg",
    "https://cdn.example/image.jpeg",
  ])("keeps optimization for remote URL %s", (src) => {
    expect(shouldBypassImageOptimization(src)).toBe(false);
  });
});
