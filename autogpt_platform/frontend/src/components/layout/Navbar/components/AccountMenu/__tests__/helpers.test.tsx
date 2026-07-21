import { IconType } from "@/components/__legacy__/ui/icons";
import { describe, expect, test } from "vitest";
import { getAccountMenuPhosphorIcon } from "../helpers";
import {
  getAccountMenuItems,
  getAccountMenuOptionIcon,
} from "../../../helpers";

function flattenTexts(groups: ReturnType<typeof getAccountMenuItems>) {
  return groups.flatMap((group) => group.items.map((item) => item.text));
}

describe("getAccountMenuPhosphorIcon", () => {
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
    const result = getAccountMenuPhosphorIcon(icon);
    expect(result).not.toBeNull();
  });

  test("returns null for unmapped icon types", () => {
    const result = getAccountMenuPhosphorIcon(IconType.Chat);
    expect(result).toBeNull();
  });
});

describe("getAccountMenuItems", () => {
  test("new layout groups profile settings and a footer without Admin for non-admins", () => {
    const groups = getAccountMenuItems(undefined, true);
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

  test("new layout adds an Admin entry for admin users", () => {
    const texts = flattenTexts(getAccountMenuItems("admin", true));

    expect(texts).toContain("Admin");
    expect(texts).toContain("Log out");
  });

  test("classic layout differs from the new layout grouping", () => {
    const classic = flattenTexts(getAccountMenuItems(undefined));
    const newLayout = flattenTexts(getAccountMenuItems(undefined, true));

    expect(classic).toContain("Profile");
    expect(newLayout).not.toEqual(classic);
  });
});

describe("getAccountMenuOptionIcon", () => {
  test.each([
    IconType.Edit,
    IconType.Settings,
    IconType.Billing,
    IconType.Help,
    IconType.WhatsNew,
    IconType.Sliders,
    IconType.LogOut,
  ])("returns an icon element for %s", (icon) => {
    expect(getAccountMenuOptionIcon(icon)).not.toBeNull();
  });

  test("falls back to a default icon for unmapped types", () => {
    expect(getAccountMenuOptionIcon(IconType.UploadCloud)).not.toBeNull();
  });
});
