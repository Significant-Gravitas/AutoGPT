import { describe, expect, it } from "vitest";

import { isProtectedOrgAvatarUrl, resolveOrgAvatarUrl } from "../avatar";

describe("resolveOrgAvatarUrl", () => {
  it("routes backend-relative avatars through the authenticated proxy", () => {
    expect(resolveOrgAvatarUrl("/api/orgs/org-1/avatar/logo.png")).toBe(
      "/api/proxy/api/orgs/org-1/avatar/logo.png",
    );
  });

  it("leaves hosted avatars unchanged", () => {
    expect(resolveOrgAvatarUrl("https://cdn.example.com/logo.png")).toBe(
      "https://cdn.example.com/logo.png",
    );
  });

  it("does not proxy an already-proxied avatar twice", () => {
    expect(
      resolveOrgAvatarUrl("/api/proxy/api/orgs/org-1/avatar/logo.png"),
    ).toBe("/api/proxy/api/orgs/org-1/avatar/logo.png");
  });

  it("only marks authenticated local avatars as protected", () => {
    expect(
      isProtectedOrgAvatarUrl("/api/proxy/api/orgs/org-1/avatar/logo.png"),
    ).toBe(true);
    expect(isProtectedOrgAvatarUrl("https://cdn.example.com/logo.png")).toBe(
      false,
    );
  });

  it("preserves an empty avatar", () => {
    expect(resolveOrgAvatarUrl(null)).toBeNull();
  });
});
