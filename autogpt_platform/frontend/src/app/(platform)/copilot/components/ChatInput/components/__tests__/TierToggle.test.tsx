import { render, screen, within } from "@/tests/integrations/test-utils";
import { expect, it, vi } from "vitest";

import { TierToggle } from "../ConnectionPicker/TierToggle";

it("exposes a locked tier as a disabled member of the radio group", () => {
  render(
    <TierToggle
      segments={[
        {
          tier: "standard",
          label: "Balanced · Sonnet",
          name: "Balanced",
          model: "Sonnet",
        },
        {
          tier: "advanced",
          label: "Advanced · Opus",
          name: "Advanced",
          model: "Opus",
          lock: { reason: "Upgrade required", href: "/profile" },
        },
      ]}
      value="standard"
      onSelect={vi.fn()}
    />,
  );

  const group = screen.getByRole("radiogroup", { name: "Model tier" });
  expect(within(group).getAllByRole("radio")).toHaveLength(2);

  const locked = within(group).getByRole("radio", { name: /Advanced · Opus/ });
  expect(locked.getAttribute("aria-disabled")).toBe("true");
  expect(locked.getAttribute("aria-checked")).toBe("false");
  expect(locked.getAttribute("tabindex")).toBe("-1");
});
