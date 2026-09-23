import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";

import { TierToggle } from "../ConnectionPicker/TierToggle";
import { MaxUpgradeCard } from "../ConnectionPicker/MaxUpgradeCard";

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

it("does not intercept navigation keys on an upgrade link or select the locked tier", async () => {
  const onSelect = vi.fn();
  render(
    <TierToggle
      segments={[
        { tier: "standard", label: "Balanced" },
        {
          tier: "advanced",
          label: "Advanced",
          lock: { reason: "Max required", href: "/settings/billing" },
        },
      ]}
      value="standard"
      onSelect={onSelect}
      advancedUpgrade={
        <MaxUpgradeCard
          label="Advanced"
          name="Advanced"
          reason="Max required"
          href="/settings/billing"
        />
      }
    />,
  );

  const balanced = screen.getByRole("radio", { name: "Balanced" });
  balanced.focus();
  await userEvent.keyboard("{ArrowDown}");
  expect(document.activeElement).toBe(balanced);
  expect(onSelect).not.toHaveBeenCalledWith("advanced");
  onSelect.mockClear();

  await userEvent.tab();
  const upgrade = screen.getByRole("link", { name: "Upgrade to Max" });
  expect(document.activeElement).toBe(upgrade);
  for (const key of [
    "ArrowDown",
    "ArrowUp",
    "ArrowLeft",
    "ArrowRight",
    "Home",
    "End",
  ]) {
    await userEvent.keyboard(`{${key}}`);
    expect(document.activeElement).toBe(upgrade);
  }
  expect(onSelect).not.toHaveBeenCalled();
});
