import { describe, expect, test } from "vitest";
import {
  getExpertAccent,
  getExpertAvatarUrl,
  getExpertFirstName,
} from "./helpers";

describe("getExpertAccent", () => {
  test("themes known roles and falls back to zinc", () => {
    expect(getExpertAccent("Marketing").pill).toContain("violet");
    expect(getExpertAccent("Sales").pill).toContain("amber");
    expect(getExpertAccent("Ops").pill).toContain("sky");
    expect(getExpertAccent("Astrologer").pill).toContain("zinc");
  });
});

describe("getExpertAvatarUrl", () => {
  test("prefers the expert's own avatar_url", () => {
    expect(
      getExpertAvatarUrl({ avatar_url: "/uploads/custom.png", role: "Sales" }),
    ).toBe("/uploads/custom.png");
  });

  test("falls back to a role-based persona avatar when avatar_url is null", () => {
    expect(getExpertAvatarUrl({ avatar_url: null, role: "Marketing" })).toBe(
      "/experts/maria.svg",
    );
    expect(
      getExpertAvatarUrl({ avatar_url: null, role: "Sales Development" }),
    ).toBe("/experts/max.svg");
    expect(getExpertAvatarUrl({ avatar_url: null, role: "Ops" })).toBe(
      "/experts/frankie.svg",
    );
  });

  test("returns null for an unplaceable role so the gradient marble shows", () => {
    expect(getExpertAvatarUrl({ avatar_url: null, role: "Astrologer" })).toBe(
      null,
    );
  });
});

describe("getExpertFirstName", () => {
  test("returns the first token of the name", () => {
    expect(getExpertFirstName("Maria Lopez")).toBe("Maria");
    expect(getExpertFirstName("Max")).toBe("Max");
    expect(getExpertFirstName("  Frankie  ")).toBe("Frankie");
  });
});
