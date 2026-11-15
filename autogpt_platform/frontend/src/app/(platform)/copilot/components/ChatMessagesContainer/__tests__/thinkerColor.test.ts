import {
  EXPERT_PALETTE,
  MANAGED_IDENTITIES,
} from "@/components/molecules/ExpertAvatar/helpers";
import { describe, expect, it } from "vitest";
import type { ExpertIdentity } from "../../../useExpertMap";
import { getThinkerHex } from "../thinkerColor";

function identity(overrides: Partial<ExpertIdentity>): ExpertIdentity {
  return {
    id: "expert-1",
    name: "Maria",
    avatarUrl: null,
    role: "SEO Content Manager",
    isArchived: false,
    readOnlyReason: null,
    ...overrides,
  };
}

describe("getThinkerHex", () => {
  it("is Otto's lavender in a thread with no expert", () => {
    expect(getThinkerHex({ expertIdentity: null })).toBe(
      EXPERT_PALETTE.otto.hex,
    );
  });

  it("is the material colour of the expert's own avatar", () => {
    const maria = MANAGED_IDENTITIES.find(
      (managed) => managed.id === "expert-maria",
    )!;

    expect(
      getThinkerHex({ expertIdentity: identity({ avatarUrl: maria.url }) }),
    ).toBe(EXPERT_PALETTE.marketing.hex);
  });

  it("is warm stone for an expert with a picture of their own", () => {
    expect(
      getThinkerHex({
        expertIdentity: identity({ avatarUrl: "https://example.com/me.png" }),
      }),
    ).toBe(EXPERT_PALETTE.general.hex);
  });

  it("stays unset while the roster is loading, rather than claim Otto", () => {
    expect(
      getThinkerHex({ expertIdentity: null, isResolvingExpertIdentity: true }),
    ).toBeNull();
  });
});
