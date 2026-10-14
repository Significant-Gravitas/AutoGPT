import {
  cleanup,
  render,
  screen,
  fireEvent,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getUsageExperience } from "@/services/usageExperience/helpers";
import { RateLimitResetDialog } from "../RateLimitResetDialog";

afterEach(cleanup);
const usage = {
  tier: "PRO",
  daily: { percent_used: 100, resets_at: "2099-10-07T00:00:00Z" },
  weekly: { percent_used: 100, resets_at: "2099-10-12T00:00:00Z" },
};

describe("Usage limit dialog", () => {
  it("offers fresh Pro usage for an exhausted lifetime trial without a reset countdown or Max", () => {
    const experience = getUsageExperience(
      { ...usage, tier: "TRIAL" },
      {
        active: true,
        converted: false,
        usage_policy: "lifetime",
        allowance_used_percent: 100,
        offer: { tier: "PRO" },
        ends_at: new Date(2026, 9, 12, 12),
      },
    );
    const onUpgrade = vi.fn();
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        experience={experience}
        offer={{ tier: "PRO", price: 5000, freshUsage: true }}
        onUpgrade={onUpgrade}
      />,
    );
    expect(screen.getByText("You’ve used your trial allowance")).toBeDefined();
    expect(
      screen.getByText("One allowance for your trial. It doesn’t refresh."),
    ).toBeDefined();
    expect(screen.queryByText(/Wait for reset|Max/)).toBeNull();
    for (const amount of screen.getAllByText(/\$50/)) {
      expect(amount.closest(".sentry-unmask, [data-sentry-unmask]")).toBeNull();
    }
    const wait = screen.getByRole("button", { name: "Wait until Oct 12" });
    expect(wait).toBeDefined();
    fireEvent.click(
      screen.getByRole("button", { name: "Start Pro now · $50" }),
    );
    expect(onUpgrade).toHaveBeenCalledOnce();
  });
  it("uses the later weekly reset when both paid limits are reached", () => {
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        experience={getUsageExperience(usage)}
      />,
    );
    expect(screen.getByText("Pro · weekly usage reached")).toBeDefined();
    expect(screen.getAllByText(/Oct 12/)[0]).toBeDefined();
    expect(screen.queryByText(/Oct 7,/)).toBeNull();
    expect(
      screen.getByRole("button", { name: "Upgrade to Max" }),
    ).toBeDefined();
  });
  it.each(["MAX", "BUSINESS", "ENTERPRISE"] as const)(
    "offers support instead of a self-service upgrade for %s",
    (tier) => {
      render(
        <RateLimitResetDialog
          isOpen
          onClose={vi.fn()}
          experience={getUsageExperience({ ...usage, tier })}
        />,
      );
      expect(screen.queryByRole("button", { name: /upgrade/i })).toBeNull();
      expect(
        screen.getByRole("link", { name: /Contact/ }).getAttribute("href"),
      ).toBe("mailto:contact@agpt.co");
    },
  );
  it("keeps linked-provider continuation actionable without discarding the blocked draft", () => {
    const onContinue = vi.fn();
    const onClose = vi.fn();
    render(
      <RateLimitResetDialog
        isOpen
        onClose={onClose}
        experience={getUsageExperience(usage)}
        alternative={{ display_name: "ChatGPT" }}
        onContinue={onContinue}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Continue on ChatGPT" }),
    );
    expect(onContinue).toHaveBeenCalledOnce();
    expect(onClose).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Upgrade to Max" }),
    ).toBeDefined();
  });
  it("does not invent an allowance when loading usage fails", () => {
    const retry = vi.fn();
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        unavailable
        onRetry={retry}
      />,
    );
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(screen.queryByText(/Upgrade to Pro/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(retry).toHaveBeenCalledOnce();
  });
  it("hides billing actions when platform payments are disabled", () => {
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        isBillingEnabled={false}
        experience={getUsageExperience(usage)}
      />,
    );
    expect(screen.queryByRole("button", { name: /upgrade/i })).toBeNull();
    expect(
      screen.getByRole("button", { name: "Wait for reset" }),
    ).toBeDefined();
  });

  it("pays the open trial invoice straight from the dialog", () => {
    const onUpgrade = vi.fn();
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        experience={getUsageExperience(
          { ...usage, tier: "NO_TIER" },
          {
            active: false,
            converted: false,
            status: "past_due",
            offer: { tier: "PRO" },
            ends_at: new Date(2026, 9, 12, 12),
            payment_due: {
              amount_due: 5000,
              currency: "usd",
              payment_url: "https://invoice.stripe.com/i/acct_1/test_inv_1",
            },
          },
        )}
        onUpgrade={onUpgrade}
      />,
    );
    expect(
      screen.getByText(
        "Pro was due to start on Oct 12, but the $50 charge didn’t go through. Pay now and Pro starts right away. Your chats, agents and results are saved.",
      ),
    ).toBeDefined();
    for (const amount of screen.getAllByText(/\$50/)) {
      expect(amount.closest(".sentry-unmask, [data-sentry-unmask]")).toBeNull();
    }
    const pay = screen.getByRole("link", { name: "Pay $50 now" });
    expect(pay.getAttribute("href")).toBe(
      "https://invoice.stripe.com/i/acct_1/test_inv_1",
    );
    expect(pay.getAttribute("target")).toBe("_blank");
    expect(
      screen.queryByRole("button", { name: "Review billing details" }),
    ).toBeNull();
    expect(onUpgrade).not.toHaveBeenCalled();
  });

  it.each([true, false])(
    "provides failed-payment recovery only when billing is enabled (%s)",
    (isBillingEnabled) => {
      const onUpgrade = vi.fn();
      render(
        <RateLimitResetDialog
          isOpen
          onClose={vi.fn()}
          isBillingEnabled={isBillingEnabled}
          experience={getUsageExperience(
            { ...usage, tier: "NO_TIER" },
            { active: false, converted: false, status: "past_due" },
          )}
          onUpgrade={onUpgrade}
        />,
      );
      expect(screen.getByText("Your payment didn’t go through")).toBeDefined();
      const recovery = screen.queryByRole("button", {
        name: "Review billing details",
      });
      if (isBillingEnabled) {
        expect(recovery).not.toBeNull();
        fireEvent.click(recovery!);
        expect(onUpgrade).toHaveBeenCalledOnce();
      } else {
        expect(recovery).toBeNull();
        expect(onUpgrade).not.toHaveBeenCalled();
      }
      expect(
        screen.queryByRole("button", { name: /^(Start|Upgrade to)/ }),
      ).toBeNull();
      expect(
        screen.getByRole("button", { name: "Back to my work" }),
      ).toBeDefined();
    },
  );
});
