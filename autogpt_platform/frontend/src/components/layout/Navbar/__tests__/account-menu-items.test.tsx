import { IconType } from "@/components/__legacy__/ui/icons";
import { describe, expect, it } from "vitest";

import { getAccountMenuPhosphorIcon } from "../components/AccountMenu/helpers";
import { getAccountMenuItems } from "../helpers";

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
});
