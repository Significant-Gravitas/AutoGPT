import { expect, test } from "vitest";
import { getExpertTopicHex } from "./colors";

test.each([
  ["Marketing", "#C47F5C"],
  ["Social & Content Repurposing", "#C47F5C"],
  ["Sales", "#C9A35B"],
  ["Operations", "#98AFC6"],
  ["Finance", "#A5B09A"],
  ["Research", "#AAA77A"],
  ["Development", "#777570"],
  ["Support", "#CB9182"],
  ["Content", "#B5ADA0"],
  ["General purpose", "#B5ADA0"],
  ["Head of AI", "#B6A4C8"],
  ["Code Review & QA", "#777570"],
])("uses the design-system color for %s", (role, hex) => {
  expect(getExpertTopicHex(role)).toBe(hex);
});

test("stored categories take priority over a broad role guess", () => {
  expect(getExpertTopicHex("Marketing writer", ["Content"])).toBe("#B5ADA0");
  expect(getExpertTopicHex("Unknown", ["other", "finance"])).toBe("#A5B09A");
});
