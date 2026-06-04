import { describe, expect, test } from "vitest";
import { getAccountMenuItems, loggedInLinks } from "../helpers";

describe("loggedInLinks", () => {
  test("does not expose Builder as a top-level navbar link", () => {
    expect(loggedInLinks.some((link) => link.href === "/build")).toBe(false);
  });
});

describe("getAccountMenuItems", () => {
  test("places the Builder entry directly after Profile", () => {
    const items = getAccountMenuItems().flatMap((group) => group.items);
    const labels = items.map((item) => item.text);
    const profileIndex = labels.indexOf("Profile");

    expect(profileIndex).toBeGreaterThanOrEqual(0);
    expect(labels[profileIndex + 1]).toBe("Builder");
  });

  test("Builder entry links to the build page", () => {
    const builder = getAccountMenuItems()
      .flatMap((group) => group.items)
      .find((item) => item.text === "Builder");

    expect(builder?.href).toBe("/build");
  });

  test("keeps Builder after Profile for admin users", () => {
    const labels = getAccountMenuItems("admin")
      .flatMap((group) => group.items)
      .map((item) => item.text);
    const profileIndex = labels.indexOf("Profile");

    expect(profileIndex).toBeGreaterThanOrEqual(0);
    expect(labels[profileIndex + 1]).toBe("Builder");
    expect(labels).toContain("Admin");
  });
});
