import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { describe, expect, test } from "vitest";
import {
  categoryForRole,
  jobTitleSuggestionsFor,
  nameSuggestionsFor,
  roleFor,
} from "./helpers";

describe("category helpers", () => {
  test("maps the wizard's role ids onto their area", () => {
    expect(categoryForRole("marketer")).toBe("marketing");
    expect(categoryForRole("developer")).toBe("development");
    expect(categoryForRole("analyst")).toBe("finance");
    expect(categoryForRole("recruiter")).toBe("operations");
  });

  test("returns null for a role outside the wizard's ids", () => {
    expect(categoryForRole("Chief of staff")).toBeNull();
  });

  test.each(Object.values(ExpertAvatarRequestCategory))(
    "gives %s a role, names and job titles",
    (category) => {
      expect(roleFor(category)).toBeTruthy();
      expect(nameSuggestionsFor(category)).not.toHaveLength(0);
      expect(jobTitleSuggestionsFor(category)).not.toHaveLength(0);
    },
  );

  test("keeps the recruiter titles under operations", () => {
    expect(jobTitleSuggestionsFor("operations")).toContain("Recruiter");
  });

  test("falls back when no area is picked", () => {
    expect(roleFor(null)).toBeNull();
    expect(nameSuggestionsFor(null)).toEqual(["Otto", "Nova", "Juno"]);
    expect(jobTitleSuggestionsFor(null)).toEqual([]);
  });
});
