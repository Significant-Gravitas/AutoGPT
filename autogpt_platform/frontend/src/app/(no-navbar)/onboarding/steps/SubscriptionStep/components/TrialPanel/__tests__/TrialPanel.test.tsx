import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { trialOffer } from "@/tests/integrations/trial-fixtures";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { formatDay } from "../helpers";
import { TrialPanel } from "../TrialPanel";

// Midday, so no time zone puts the dates below on another calendar day.
const NOW = new Date(2026, 9, 27, 12, 0, 0);

const OFFER: TrialOfferResponse = {
  ...trialOffer,
  duration_days: 7,
  unit_amount: 5000,
  onboarding_credit_amount: 2000,
};

function renderTrial(
  overrides: Partial<ComponentProps<typeof TrialPanel>> = {},
) {
  const onStart = vi.fn();
  render(
    <TrialPanel
      offer={OFFER}
      isStarting={false}
      isBlocked={false}
      error={null}
      onStart={onStart}
      {...overrides}
    />,
  );
  return { onStart };
}

function getRows() {
  const timeline = screen.getByRole("list", { name: "How your trial works" });
  return within(timeline).getAllByRole("listitem");
}

function getStartButton() {
  return screen.getByRole("button", { name: "Start my free trial" });
}

beforeEach(() => {
  vi.setSystemTime(NOW);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("TrialPanel", () => {
  test("introduces the trial and what it costs today", () => {
    renderTrial();

    expect(screen.getByText("7-day free trial")).toBeDefined();
    expect(
      screen.getByRole("heading", {
        level: 1,
        name: "Your first 7 days are on us",
      }),
    ).toBeDefined();
    expect(
      screen.getByText(
        "Nothing to pay today, and a reminder before there's ever anything to pay.",
      ),
    ).toBeDefined();
  });

  test("starts the trial today with the usage it includes", () => {
    renderTrial();

    const rows = getRows();
    expect(rows).toHaveLength(3);
    const [today] = rows;
    expect(within(today).getByText("Today")).toBeDefined();
    expect(within(today).getByText("Your free trial starts")).toBeDefined();
    expect(
      within(today).getByText(
        "Full access from right now, with $20 of usage included.",
      ),
    ).toBeDefined();
  });

  test("falls back to a plain allowance line when no credit comes with it", () => {
    renderTrial({ offer: { ...OFFER, onboarding_credit_amount: 0 } });

    expect(
      within(getRows()[0]).getByText(
        "Full access from right now, within a trial usage allowance.",
      ),
    ).toBeDefined();
  });

  test("emails a reminder three days before the first payment", () => {
    renderTrial();

    const reminder = getRows()[1];
    const when = within(reminder).getByText(formatDay(new Date(2026, 9, 31)));
    expect(when.tagName).toBe("TIME");
    expect(when.getAttribute("datetime")).toBe("2026-10-31");
    expect(within(reminder).getByText("We email you a reminder")).toBeDefined();
    expect(
      within(reminder).getByText(
        "3 days before your trial ends, with your end date and how to cancel.",
      ),
    ).toBeDefined();
  });

  test("charges when the trial ends and promises the day before to cancel", () => {
    renderTrial();

    const charge = getRows()[2];
    const when = within(charge).getByText(formatDay(new Date(2026, 10, 3)));
    expect(when.tagName).toBe("TIME");
    expect(when.getAttribute("datetime")).toBe("2026-11-03");
    expect(within(charge).getByText("Your first payment: $50")).toBeDefined();
    expect(
      within(charge).getByText(
        `Then $50 a month, plus any applicable tax. Cancel by ${formatDay(new Date(2026, 10, 2))} to pay nothing.`,
      ),
    ).toBeDefined();
  });

  test("starts the trial from its one button", async () => {
    const user = userEvent.setup();
    const { onStart } = renderTrial();

    await user.click(getStartButton());

    expect(onStart).toHaveBeenCalledTimes(1);
  });

  test("holds the button while the trial is starting", async () => {
    const user = userEvent.setup();
    const { onStart } = renderTrial({ isStarting: true });

    const button = getStartButton() as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    await user.click(button);
    expect(onStart).not.toHaveBeenCalled();
  });

  test("holds the button while the step is blocked", async () => {
    const user = userEvent.setup();
    const { onStart } = renderTrial({ isBlocked: true });

    const button = getStartButton() as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    await user.click(button);
    expect(onStart).not.toHaveBeenCalled();
  });

  test("shows the error under the button", () => {
    renderTrial({ error: "We couldn't reach Stripe. Try again." });

    expect(screen.getByRole("alert").textContent).toBe(
      "We couldn't reach Stripe. Try again.",
    );
  });

  test("shows no error line without an error", () => {
    renderTrial();

    expect(screen.queryByRole("alert")).toBeNull();
  });

  test("says a card is asked for but nothing is charged", () => {
    renderTrial();

    expect(
      screen.getByText("Stripe will ask for a card. Nothing is charged today."),
    ).toBeDefined();
  });
});
