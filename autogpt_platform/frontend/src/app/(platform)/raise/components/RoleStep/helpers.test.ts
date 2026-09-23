import { describe, expect, test } from "vitest";
import {
  aboutPlaceholderFor,
  CUSTOM_ROLE_MAX_LENGTH,
  isValidCustomRole,
  normalizeCustomRole,
  roleLabelFor,
  roleOptionsForSelection,
} from "./helpers";

describe("RoleStep helpers", () => {
  test("roleLabelFor returns preset labels", () => {
    expect(roleLabelFor("marketer")).toBe("Marketer");
  });

  test("roleLabelFor returns custom role text", () => {
    expect(roleLabelFor("UX Designer")).toBe("UX Designer");
  });

  test("roleOptionsForSelection collapses to a custom role chip", () => {
    expect(roleOptionsForSelection("UX Designer")).toEqual([
      { id: "UX Designer", label: "UX Designer" },
    ]);
  });

  test("normalizeCustomRole trims whitespace", () => {
    expect(normalizeCustomRole("  Coach  ")).toBe("Coach");
  });

  test("isValidCustomRole rejects empty and overlong values", () => {
    expect(isValidCustomRole("")).toBe(false);
    expect(isValidCustomRole("   ")).toBe(false);
    expect(isValidCustomRole("a".repeat(CUSTOM_ROLE_MAX_LENGTH))).toBe(true);
    expect(isValidCustomRole("a".repeat(CUSTOM_ROLE_MAX_LENGTH + 1))).toBe(
      false,
    );
  });

  test("aboutPlaceholderFor returns a generic placeholder without a name", () => {
    expect(aboutPlaceholderFor(null)).toBe(
      "How they should work, what you care about, anything that helps them sound like yours…",
    );
    expect(aboutPlaceholderFor("   ")).toBe(
      "How they should work, what you care about, anything that helps them sound like yours…",
    );
  });

  test("aboutPlaceholderFor personalizes the placeholder with the expert name", () => {
    expect(aboutPlaceholderFor("Nova")).toBe(
      "How Nova should work, what you care about, anything that helps them sound like yours…",
    );
    expect(aboutPlaceholderFor("  Ada  ")).toBe(
      "How Ada should work, what you care about, anything that helps them sound like yours…",
    );
  });
});
