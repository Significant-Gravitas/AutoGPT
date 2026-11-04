import { describe, expect, test } from "vitest";
import { normalizeOnboardingProfile } from "../helpers";

describe("normalizeOnboardingProfile", () => {
  test("cuts an Other role to the 100 code points the profile accepts", () => {
    // The API counts code points, so 100 emoji fit; a UTF-16 cut keeps 50.
    const { role } = normalizeOnboardingProfile({
      role: "Other",
      otherRole: "🙂".repeat(150),
      painPoints: [],
      otherPainPoint: "",
    });
    expect(role).toBe("🙂".repeat(100));
  });
});
