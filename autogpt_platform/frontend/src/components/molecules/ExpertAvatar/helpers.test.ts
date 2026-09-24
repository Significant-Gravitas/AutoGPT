import { describe, expect, it } from "vitest";
import { resolveExpertAvatarUrl } from "./helpers";

describe("expert PNG avatars", () => {
  it("maps stored roster URLs without depending on the current name", () => {
    expect(resolveExpertAvatarUrl("/experts/maria.svg")).toBe(
      "/autogpt-characters/v1.1/expert-maria/neutral/128.webp",
    );
  });

  it("preserves uploads and saved PNG choices", () => {
    for (const url of [
      "https://cdn.example/upload.png",
      "/api/store/media/user/images/custom.png",
      "/experts/clay/v1/finance.png",
    ]) {
      expect(resolveExpertAvatarUrl(url)).toBe(url);
    }
  });

  it("uses a warm-stone default for missing and custom legacy avatars", () => {
    expect(resolveExpertAvatarUrl(null)).toBe("/experts/clay/v1/content.png");
    expect(resolveExpertAvatarUrl("/avatars/notion/1-2-3.violet.svg")).toBe(
      "/experts/clay/v1/content.png",
    );
  });
});
