import { describe, expect, test } from "vitest";
import { getExpertRoleLabel } from "./expert-role-label";

describe("getExpertRoleLabel", () => {
  test("shortens long roster roles", () => {
    expect(getExpertRoleLabel("Social & Content Repurposing")).toBe(
      "Social media",
    );
    expect(getExpertRoleLabel("Market & Competitor Intelligence")).toBe(
      "Market intelligence",
    );
  });

  test("keeps other roles unchanged", () => {
    expect(getExpertRoleLabel("Email & Lifecycle")).toBe("Email & Lifecycle");
  });
});
