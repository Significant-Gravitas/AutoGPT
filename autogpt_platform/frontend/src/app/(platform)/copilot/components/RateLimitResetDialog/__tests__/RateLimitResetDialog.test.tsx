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
      },
    );
    const onUpgrade = vi.fn();
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        experience={experience}
        onUpgrade={onUpgrade}
      />,
    );
    expect(screen.getByText("Keep your momentum.")).toBeDefined();
    expect(
      screen.getByText("One allowance for your trial. It doesn’t refresh."),
    ).toBeDefined();
    expect(screen.queryByText(/Wait for reset|Max/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Upgrade to Pro" }));
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
    expect(screen.getByText(/Oct 12/)).toBeDefined();
    expect(screen.queryByText(/Oct 7,/)).toBeNull();
    expect(
      screen.getByRole("button", { name: "Review Max upgrade" }),
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
      screen.getByRole("button", { name: "Review Max upgrade" }),
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
  it("leads a top-tier plan with the reset, the meters and Send at reset", () => {
    const onSendAtReset = vi.fn();
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        experience={getUsageExperience({ ...usage, tier: "MAX" })}
        wait={{ window: "weekly", resetsAt: new Date(Date.now() + 3e8) }}
        meters={usage}
        onSendAtReset={onSendAtReset}
      />,
    );
    expect(
      screen.getByText("You’re out of AutoPilot budget for this week"),
    ).toBeDefined();
    expect(screen.queryByText("A little pause. Everything is saved.")).toBe(
      null,
    );
    expect(screen.getByText(/^It resets \w+ at .+, in 3 days\./)).toBeDefined();
    expect(screen.getAllByRole("progressbar")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "Send at reset" }));
    expect(onSendAtReset).toHaveBeenCalledOnce();
    expect(screen.getByRole("link", { name: "Contact us" })).toBeDefined();
  });
  it("keeps the upgrade as the lead for other plans and adds the exact reset", () => {
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        experience={getUsageExperience({ ...usage, weekly: undefined })}
        wait={{ window: "daily", resetsAt: new Date(Date.now() + 6.03e6) }}
        meters={usage}
        onSendAtReset={vi.fn()}
      />,
    );
    expect(screen.getByText("More room for what’s next.")).toBeDefined();
    expect(screen.getByText(/^It resets at .+, in 1 h 40 min\./)).toBeDefined();
    expect(screen.getByRole("button", { name: "Send at reset" })).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Review Max upgrade" }),
    ).toBeDefined();
  });
  it("offers no waiting when the window allows nothing", () => {
    render(
      <RateLimitResetDialog
        isOpen
        onClose={vi.fn()}
        experience={getUsageExperience({
          ...usage,
          tier: "NO_TIER",
          state: "exhausted",
          limiting_window: "weekly",
          resets_at: null,
        })}
      />,
    );
    expect(screen.queryByText(/You can continue after|resets/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Send at reset" })).toBeNull();
    expect(
      screen.getByRole("button", { name: "Upgrade to Pro" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Back to my work" }),
    ).toBeDefined();
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
});
