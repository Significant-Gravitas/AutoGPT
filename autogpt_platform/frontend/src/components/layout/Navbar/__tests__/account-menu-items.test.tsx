import { IconRefresh, IconType } from "@/components/__legacy__/ui/icons";
import { describe, expect, it } from "vitest";

import { getAccountMenuIcon } from "../components/AccountMenu/helpers";
import { getAccountMenuItems, getAccountMenuOptionIcon } from "../helpers";

describe("account menu changelog entry", () => {
  it("includes a 'Changelog' link out to the public changelog, above Help & Docs", () => {
    const items = getAccountMenuItems().flatMap((group) => group.items);
    const changelog = items.find((item) => item.text === "Changelog");

    expect(changelog).toBeDefined();
    expect(changelog?.external).toBe(true);
    expect(changelog?.href).toBe("https://agpt.co/changelog");
    expect(changelog?.icon).toBe(IconType.WhatsNew);

    const changelogIndex = items.findIndex((i) => i.text === "Changelog");
    const helpIndex = items.findIndex((i) => i.text === "Help & Docs");
    expect(changelogIndex).toBeLessThan(helpIndex);
  });

  it("maps the Changelog icon in the account menu", () => {
    expect(getAccountMenuIcon(IconType.WhatsNew)).not.toBeNull();
  });

  it("maps the Changelog icon in the mobile menu (not the fallback)", () => {
    const changelogIcon = getAccountMenuOptionIcon(IconType.WhatsNew);
    const helpIcon = getAccountMenuOptionIcon(IconType.Help);

    expect(changelogIcon.type).not.toBe(IconRefresh);
    expect(changelogIcon.props.icon).not.toBe(helpIcon.props.icon);
  });
});
