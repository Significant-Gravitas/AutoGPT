import { expect, test } from "vitest";
import { getCategoryHex, getExpertTopicHex } from "./colors";

const MARIA = "/autogpt-characters/v1.1/expert-maria/neutral/128.webp";

test.each([
  ["marketing", "#C47F5C"],
  ["sales", "#C9A35B"],
  ["finance", "#A5B09A"],
  ["support", "#CB9182"],
  ["operations", "#98AFC6"],
  ["research", "#AAA77A"],
  ["content", "#81AAA6"],
  ["development", "#777570"],
  ["general", "#B5ADA0"],
  ["otto", "#B6A4C8"],
])("uses the design-system material anchor for %s", (category, hex) => {
  expect(getCategoryHex(category)).toBe(hex);
  expect(getCategoryHex(category.toUpperCase())).toBe(hex);
});

test("a managed identity keeps its own family whatever the filter or category edit says", () => {
  expect(getExpertTopicHex({ avatarUrl: MARIA, categories: ["content"] })).toBe(
    "#C47F5C",
  );
  expect(
    getExpertTopicHex({
      avatarUrl: "/experts/clay/v4/quinn-finance.png",
      categories: ["finance", "research"],
    }),
  ).toBe("#AAA77A");
});

test("a custom appearance takes the first stored category and General without one", () => {
  expect(
    getExpertTopicHex({
      avatarUrl: "https://cdn.test/mine.png",
      categories: ["Support", "operations"],
    }),
  ).toBe("#CB9182");
  expect(
    getExpertTopicHex({
      avatarUrl: "https://cdn.test/mine.png",
      categories: ["other", "finance"],
    }),
  ).toBe("#A5B09A");
  expect(
    getExpertTopicHex({ avatarUrl: null, categories: [], role: "Marketing" }),
  ).toBe("#B5ADA0");
  expect(getExpertTopicHex({ role: "Head of AI" })).toBe("#B6A4C8");
});

test("the role never picks a specialist color", () => {
  expect(getExpertTopicHex({ role: "Sales coach", categories: null })).toBe(
    "#B5ADA0",
  );
  expect(getCategoryHex("Sales coach")).toBeUndefined();
});
