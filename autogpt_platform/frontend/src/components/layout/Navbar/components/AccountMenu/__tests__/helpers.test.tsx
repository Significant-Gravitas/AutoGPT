import { IconType } from "@/components/__legacy__/ui/icons";
import { describe, expect, test } from "vitest";
import { getAccountMenuPhosphorIcon } from "../helpers";

describe("getAccountMenuPhosphorIcon", () => {
  test.each([
    IconType.Edit,
    IconType.LayoutDashboard,
    IconType.UploadCloud,
    IconType.Sliders,
    IconType.Settings,
    IconType.Billing,
    IconType.Help,
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
