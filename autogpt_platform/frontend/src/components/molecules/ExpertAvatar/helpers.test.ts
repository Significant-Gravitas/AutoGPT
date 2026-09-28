import { describe, expect, it } from "vitest";
import {
  DEFAULT_EXPERT_AVATAR_URL,
  getExpertVisualCategory,
  getManagedAvatar,
  getManagedIdentity,
  MANAGED_IDENTITIES,
  resolveExpertAvatarUrl,
} from "./helpers";

const MARIA = "/autogpt-characters/v1.1/expert-maria/neutral/128.webp";
const JULES = "/autogpt-characters/v2.1/expert-jules/neutral/128.webp";

describe("managed expert identities", () => {
  it("maps stored roster and retired clay defaults to the identity they stood for", () => {
    expect(resolveExpertAvatarUrl("/experts/maria.svg")).toBe(MARIA);
    expect(resolveExpertAvatarUrl("/experts/clay/v5/jules-marketing.png")).toBe(
      JULES,
    );
    expect(resolveExpertAvatarUrl("/experts/clay/v4/jules-content.png")).toBe(
      JULES,
    );
    expect(
      resolveExpertAvatarUrl(
        "/avatars/notion/12-5-13-13-3-9-2-11-0-0.fuchsia.svg",
      ),
    ).toBe(JULES);
  });

  it("preserves uploads, generated images and managed URLs", () => {
    for (const url of [
      "https://cdn.example/upload.png",
      "/api/store/media/user/images/custom.png",
      "/avatars/mine.svg",
      MARIA,
      JULES,
      DEFAULT_EXPERT_AVATAR_URL,
    ]) {
      expect(resolveExpertAvatarUrl(url)).toBe(url);
    }
  });

  it("uses the General fallback for missing avatars, custom Notion picks and the retired category sheets", () => {
    expect(DEFAULT_EXPERT_AVATAR_URL).toBe(
      "/autogpt-characters/v2.1/expert-general-01/neutral/128.webp",
    );
    expect(resolveExpertAvatarUrl(null)).toBe(DEFAULT_EXPERT_AVATAR_URL);
    expect(resolveExpertAvatarUrl("/avatars/notion/1-2-3.violet.svg")).toBe(
      DEFAULT_EXPERT_AVATAR_URL,
    );
    expect(resolveExpertAvatarUrl("/experts/clay/v1/finance.png")).toBe(
      DEFAULT_EXPERT_AVATAR_URL,
    );
  });

  it("serves every identity from its own versioned library path", () => {
    expect(MANAGED_IDENTITIES).toHaveLength(34);
    for (const identity of MANAGED_IDENTITIES) {
      expect(identity.url).toBe(
        `${identity.base_url}/${identity.id}/neutral/128.webp`,
      );
      expect(getManagedIdentity(identity.url)?.id).toBe(identity.id);
    }
    expect(getManagedAvatar(JULES, 40)).toMatchObject({
      assetID: "expert-jules",
      base: "/autogpt-characters/v2.1/expert-jules/neutral",
      pixels: 40,
    });
    expect(getManagedAvatar(MARIA, 88)?.pixels).toBe(96);
    expect(getManagedAvatar("https://cdn.example/upload.png", 40)).toBeNull();
    expect(
      getManagedAvatar(
        "/autogpt-characters/v9.9/expert-jules/neutral/128.webp",
        40,
      ),
    ).toBeNull();
  });

  it("keeps the visual family with the identity, not the filter", () => {
    expect(getExpertVisualCategory(MARIA, ["content"])).toBe("marketing");
    expect(getExpertVisualCategory("https://cdn.test/x.png", ["Sales"])).toBe(
      "sales",
    );
    expect(
      getExpertVisualCategory(DEFAULT_EXPERT_AVATAR_URL, ["research"]),
    ).toBe("research");
    expect(getExpertVisualCategory(null, ["otto"])).toBe("general");
    expect(getExpertVisualCategory(null, null)).toBe("general");
  });
});
