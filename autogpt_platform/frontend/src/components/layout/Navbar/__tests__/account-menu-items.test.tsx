import { IconType } from "@/components/__legacy__/ui/icons";
import { describe, expect, it } from "vitest";

import { getAccountMenuPhosphorIcon } from "../components/AccountMenu/helpers";
import { getAccountMenuItems } from "../helpers";

describe("account menu changelog entry", () => {
  it("includes a 'What's New' link out to the public changelog", () => {
    const items = getAccountMenuItems().flatMap((group) => group.items);
    const whatsNew = items.find((item) => item.text === "What's New");

    expect(whatsNew).toBeDefined();
    expect(whatsNew?.external).toBe(true);
    expect(whatsNew?.href).toContain("/platform/changelog/");
    expect(whatsNew?.icon).toBe(IconType.Changelog);
  });

  it("maps the Changelog icon to a phosphor icon", () => {
    expect(getAccountMenuPhosphorIcon(IconType.Changelog)).not.toBeNull();
  });
});
