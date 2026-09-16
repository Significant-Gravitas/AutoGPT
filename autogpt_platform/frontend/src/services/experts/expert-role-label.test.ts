import { describe, expect, test } from "vitest";
import { getExpertRoleLabel } from "./expert-role-label";

describe("getExpertRoleLabel", () => {
  test("shortens long roster roles", () => {
    expect(getExpertRoleLabel("Social & Content Repurposing")).toBe(
      "Social Media",
    );
    expect(getExpertRoleLabel("Market & Competitor Intelligence")).toBe(
      "Market Intelligence",
    );
  });

  test("keeps other roles unchanged", () => {
    expect(getExpertRoleLabel("Email & Lifecycle")).toBe("Email & Lifecycle");
  });
});
