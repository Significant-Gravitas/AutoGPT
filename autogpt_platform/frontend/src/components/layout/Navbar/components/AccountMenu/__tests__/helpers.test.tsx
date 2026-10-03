import { IconType } from "@/components/__legacy__/ui/icons";
import { describe, expect, test } from "vitest";
import { getAccountMenuIcon } from "../helpers";
import { getAccountMenuItems } from "../../../helpers";

function flattenTexts(groups: ReturnType<typeof getAccountMenuItems>) {
  return groups.flatMap((group) => group.items.map((item) => item.text));
}

describe("getAccountMenuIcon", () => {
  test.each([
    IconType.Edit,
    IconType.LayoutDashboard,
    IconType.UploadCloud,
    IconType.Sliders,
    IconType.Settings,
    IconType.Billing,
    IconType.Help,
    IconType.WhatsNew,
    IconType.LogOut,
  ])("returns a Phosphor icon element for %s", (icon) => {
    const result = getAccountMenuIcon(icon);
    expect(result).not.toBeNull();
  });

  test("returns null for unmapped icon types", () => {
    const result = getAccountMenuIcon(IconType.Chat);
    expect(result).toBeNull();
  });
});

describe("getAccountMenuItems", () => {
  test("groups profile settings and a footer without Admin for non-admins", () => {
    const groups = getAccountMenuItems(undefined);
    const texts = flattenTexts(groups);

    expect(texts).toEqual(
      expect.arrayContaining([
        "Profile",
        "Settings",
        "Billing",
        "What's new",
        "Help & Docs",
        "Log out",
      ]),
    );
    expect(texts).not.toContain("Admin");
  });

  test("points What's new at the changelog docs", () => {
    const items = getAccountMenuItems(undefined).flatMap(
      (group) => group.items,
    );
    const whatsNew = items.find((item) => item.text === "What's new");

    expect(whatsNew?.href).toBe(
      "https://agpt.co/docs/platform/changelog/changelog/",
    );
    expect(whatsNew?.external).toBe(true);
  });

  test("adds an Admin entry for admin users", () => {
    const texts = flattenTexts(getAccountMenuItems("admin"));

    expect(texts).toContain("Admin");
    expect(texts).toContain("Log out");
  });
});
