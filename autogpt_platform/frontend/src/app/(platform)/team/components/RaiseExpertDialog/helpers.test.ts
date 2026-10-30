import { describe, expect, test } from "vitest";
import { inferArea, jobTitleForRoleLink } from "./helpers";

describe("inferArea", () => {
  test.each([
    ["Deps merger", "", "development"],
    ["Ada", "Release engineer", "development"],
    ["Pitch", "Sales analyst", "sales"],
    ["Tally", "Invoice chaser", "finance"],
    ["Inbox zero", "Support specialist", "support"],
    ["Quill", "Newsletter writer", "content"],
    ["Kepler", "Competitor watcher", "research"],
    ["Cadence", "Recruiting coordinator", "operations"],
    ["Echo", "SEO lead", "marketing"],
  ])("%s / %s works in %s", (name, role, category) => {
    expect(inferArea(name, role).category).toBe(category);
  });

  test("falls back to general with its colour and role", () => {
    expect(inferArea("Juno", "")).toEqual({
      category: "general",
      role: "Assistant",
      color: "amber-300",
    });
  });
});

describe("jobTitleForRoleLink", () => {
  test("prefills the role from Otto's suggestion links", () => {
    expect(jobTitleForRoleLink("developer")).toBe("Software engineer");
    expect(jobTitleForRoleLink("unknown")).toBe("");
    expect(jobTitleForRoleLink(null)).toBe("");
  });
});
