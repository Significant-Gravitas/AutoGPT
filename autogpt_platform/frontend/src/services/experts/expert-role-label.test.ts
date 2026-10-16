import { describe, expect, test } from "vitest";
import { getExpertRoleLabel } from "./expert-role-label";

describe("getExpertRoleLabel", () => {
  test("shortens long roster roles", () => {
    expect(getExpertRoleLabel("Social & Content Repurposing")).toBe(
      "Social Media Manager",
    );
    expect(getExpertRoleLabel("Market & Competitor Intelligence")).toBe(
      "Market Research Analyst",
    );
  });

  test("keeps other roles unchanged", () => {
    expect(getExpertRoleLabel("My custom role")).toBe("My custom role");
  });
});
