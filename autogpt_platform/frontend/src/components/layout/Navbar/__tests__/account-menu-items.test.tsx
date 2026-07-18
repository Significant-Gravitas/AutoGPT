import { IconRefresh, IconType } from "@/components/__legacy__/ui/icons";
import { describe, expect, it } from "vitest";

import { getAccountMenuPhosphorIcon } from "../components/AccountMenu/helpers";
import { getAccountMenuItems, getAccountMenuOptionIcon } from "../helpers";

describe("account menu changelog entry", () => {
  it("includes a 'Changelog' link out to the public changelog, above Help & Docs", () => {
    const items = getAccountMenuItems().flatMap((group) => group.items);
    const changelog = items.find((item) => item.text === "Changelog");

    expect(changelog).toBeDefined();
    expect(changelog?.external).toBe(true);
    expect(changelog?.href).toBe(
      "https://agpt.co/docs/platform/changelog/changelog/",
    );
    expect(changelog?.icon).toBe(IconType.Changelog);

    const changelogIndex = items.findIndex((i) => i.text === "Changelog");
    const helpIndex = items.findIndex((i) => i.text === "Help & Docs");
    expect(changelogIndex).toBeLessThan(helpIndex);
  });

  it("maps the Changelog icon to a phosphor icon", () => {
    expect(getAccountMenuPhosphorIcon(IconType.Changelog)).not.toBeNull();
  });

  it("maps the Changelog icon in the mobile menu (not the fallback)", () => {
    // Mobile renders via getAccountMenuOptionIcon; it must handle Changelog
    // explicitly rather than falling through to the default refresh icon.
    const changelogIcon = getAccountMenuOptionIcon(IconType.Changelog);
    const helpIcon = getAccountMenuOptionIcon(IconType.Help);

    expect(changelogIcon.type).not.toBe(IconRefresh);
    expect(changelogIcon.type).not.toBe(helpIcon.type);
  });
});
