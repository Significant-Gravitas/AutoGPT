import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { formatDay } from "../helpers";
import { PaidPanel } from "../PaidPanel";

// Midday, so no time zone puts the dates below on another calendar day.
const NOW = new Date(2026, 9, 27, 12, 0, 0);

function renderPaid(overrides: Partial<ComponentProps<typeof PaidPanel>> = {}) {
  const onContinue = vi.fn();
  render(
    <PaidPanel
      price="$50"
      cadence="month"
      isProcessing={false}
      isBlocked={false}
      onContinue={onContinue}
      {...overrides}
    />,
  );
  return { onContinue };
}

function getRows() {
  const timeline = screen.getByRole("list", { name: "How your plan works" });
  return within(timeline).getAllByRole("listitem");
}

function getCheckoutButton() {
  return screen.getByRole("button", { name: "Continue to checkout" });
}

beforeEach(() => {
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("PaidPanel", () => {
  test("introduces Pro and its price", () => {
    renderPaid();

    expect(screen.getByText("Pro plan")).toBeDefined();
    expect(
      screen.getByRole("heading", { level: 1, name: "Start with Pro" }),
    ).toBeDefined();
    expect(
      screen.getByText(
        "$50 a month, plus any applicable tax. No contract, and you can cancel any time in Settings.",
      ),
    ).toBeDefined();
  });

  test("starts the plan today and renews it a month later", () => {
    renderPaid();

    const rows = getRows();
    expect(rows).toHaveLength(2);
    const [today, renews] = rows;

    expect(within(today).getByText("Today")).toBeDefined();
    expect(within(today).getByText("Your plan starts")).toBeDefined();
    expect(
      within(today).getByText(
        "You pay $50, plus any applicable tax. Full access starts straight away.",
      ),
    ).toBeDefined();

    const when = within(renews).getByText(formatDay(new Date(2026, 10, 27)));
    expect(when.tagName).toBe("TIME");
    expect(when.getAttribute("datetime")).toBe("2026-11-27");
    expect(within(renews).getByText("Your plan renews")).toBeDefined();
    expect(
      within(renews).getByText("$50 a month until you cancel in Settings."),
    ).toBeDefined();
  });

  test("renews a yearly plan a year later", () => {
    renderPaid({ price: "$480", cadence: "year" });

    expect(
      screen.getByText(
        "$480 a year, plus any applicable tax. No contract, and you can cancel any time in Settings.",
      ),
    ).toBeDefined();
    const renews = getRows()[1];
    const when = within(renews).getByText(formatDay(new Date(2027, 9, 27)));
    expect(when.getAttribute("datetime")).toBe("2027-10-27");
    expect(
      within(renews).getByText("$480 a year until you cancel in Settings."),
    ).toBeDefined();
  });

  test("continues to checkout from its one button", async () => {
    const user = userEvent.setup();
    const { onContinue } = renderPaid();

    await user.click(getCheckoutButton());

    expect(onContinue).toHaveBeenCalledTimes(1);
  });

  test("holds the button while checkout is opening", async () => {
    const user = userEvent.setup();
    const { onContinue } = renderPaid({ isProcessing: true });

    const button = getCheckoutButton() as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    await user.click(button);
    expect(onContinue).not.toHaveBeenCalled();
  });

  test("holds the button while the step is blocked", async () => {
    const user = userEvent.setup();
    const { onContinue } = renderPaid({ isBlocked: true });

    const button = getCheckoutButton() as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    await user.click(button);
    expect(onContinue).not.toHaveBeenCalled();
  });

  test("shows the notice it is given", () => {
    renderPaid({
      notice: <p>Your last checkout didn&apos;t go through.</p>,
    });

    expect(
      screen.getByText("Your last checkout didn't go through."),
    ).toBeDefined();
  });

  test("shows no error line", () => {
    renderPaid();

    expect(screen.queryByRole("alert")).toBeNull();
  });

  test("names Stripe as the checkout", () => {
    renderPaid();

    expect(screen.getByText("Secure checkout by Stripe")).toBeDefined();
  });
});
