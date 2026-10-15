import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import { UsageLimitReachedCard } from "../UsageLimitReachedCard";
import {
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
const provider = vi.hoisted(() => ({ use: vi.fn(), continueHere: vi.fn() }));
vi.mock("../../../ProviderLimitDialog/useProviderLimitDialog", () => ({
  useProviderLimitDialog: provider.use,
}));
beforeEach(() =>
  provider.use.mockReturnValue({
    alternative: null,
    continueHere: provider.continueHere,
    isSwitching: false,
  }),
);
vi.mock("@/services/pro-activation/useProActivation", () => ({
  useProActivation: () => ({ start: vi.fn(), isBusy: false, isReady: false }),
}));
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => true,
}));
afterEach(() => {
  cleanup();
  setTrialUser(null);
});
function setup(tier = "PRO", daily = 100, weekly = 40) {
  server.use(
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({
        tier,
        daily: { percent_used: daily, resets_at: "2099-10-07T00:00:00Z" },
        weekly: { percent_used: weekly, resets_at: "2099-10-12T00:00:00Z" },
      }),
    ),
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({
        tier,
        tier_costs: { PRO: 5000, MAX: 32000 },
        monthly_cost: 5000,
        proration_credit_cents: 0,
      }),
    ),
  );
}
describe("Usage notice", () => {
  it("shows the accepted trial price and its automatic Pro start date", async () => {
    setTrialUser();
    setup("TRIAL");
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json(
          trialResponse({
            allowance_used_percent: 100,
            usage_policy: "lifetime",
            ends_at: new Date(2026, 9, 12, 12),
            offer: { ...trialOffer, unit_amount: 5013 },
          }),
        ),
      ),
    );
    render(<UsageLimitReachedCard />);
    expect(
      await screen.findByRole("button", { name: "Start Pro now · $50.13" }),
    ).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "Start Pro now · $50.13" })
        .closest(".sentry-unmask, [data-sentry-unmask]"),
    ).toBeNull();
    expect(screen.getByText("You’ve used your trial allowance")).toBeDefined();
    expect(
      screen.getByText(
        "Pro starts on its own on Oct 12. Start it now to keep working today.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("link", { name: "View usage" })).toBeDefined();
  });
  it("continues on a linked provider only after an explicit click", async () => {
    setup();
    provider.use.mockReturnValue({
      alternative: { display_name: "ChatGPT" },
      continueHere: provider.continueHere,
      isSwitching: false,
    });
    render(<UsageLimitReachedCard sessionID="chat" />);
    const action = await screen.findByRole("button", {
      name: "Continue on ChatGPT",
    });
    expect(provider.continueHere).not.toHaveBeenCalled();
    fireEvent.click(action);
    expect(provider.continueHere).toHaveBeenCalledOnce();
  });
  it("prioritizes the blocking window and preserves draft editing", async () => {
    setup("PRO", 100, 100);
    render(<UsageLimitReachedCard />);
    expect(
      await screen.findByText("You’ve hit this week’s limit"),
    ).toBeDefined();
    expect(screen.getByText(/Oct 12/)).toBeDefined();
    expect(screen.getByText("You can keep editing your draft.")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Upgrade to Max" }),
    ).toBeDefined();
  });
  it("does not promote an unavailable self-service tier to Enterprise", async () => {
    setup("ENTERPRISE");
    render(<UsageLimitReachedCard />);
    expect(
      await screen.findByRole("link", { name: "Contact your account team" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /upgrade/i })).toBeNull();
  });
  it("hides the limit notice while allowance remains", async () => {
    setup("PRO", 10, 20);
    const { container } = render(<UsageLimitReachedCard />);
    await waitFor(() => expect(container.innerHTML).toBe(""));
  });
  it("offers the bounded refusal retry even when cached usage is below the cap", async () => {
    setup("PRO", 10, 20);
    const refresh = vi.fn().mockResolvedValue(undefined);
    const { rerender } = render(
      <UsageLimitReachedCard
        refresh={{ checking: false, failed: true, refresh }}
      />,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    expect(refresh).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: /upgrade/i })).toBeNull();
    rerender(
      <UsageLimitReachedCard
        refresh={{ checking: true, failed: false, refresh }}
      />,
    );
    expect(screen.getByText("Refreshing your plan and usage…")).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "Checking…" })
        .hasAttribute("disabled"),
    ).toBe(true);
    rerender(
      <UsageLimitReachedCard
        refresh={{ checking: false, failed: false, refresh }}
      />,
    );
    await waitFor(() =>
      expect(screen.queryByRole("status", { name: "Usage notice" })).toBeNull(),
    );
  });
  it("lets a failed usage fetch be retried without showing invented remaining usage", async () => {
    setup();
    server.use(
      http.get(
        "*/api/chat/usage",
        () => new HttpResponse(null, { status: 503 }),
      ),
    );
    render(<UsageLimitReachedCard />);
    expect(
      await screen.findByText(
        "We couldn’t check your usage.",
        {},
        { timeout: 5000 },
      ),
    ).toBeDefined();
    setup();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("You’ve hit today’s limit")).toBeDefined();
  });
  it("pays a failed first trial charge from the notice", async () => {
    setTrialUser();
    setup("NO_TIER");
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json(
          trialResponse({
            active: false,
            status: "past_due",
            ends_at: new Date(2026, 9, 12, 12),
            payment_due: {
              amount_due: 5000,
              currency: "usd",
              payment_url: "https://invoice.stripe.com/i/acct_1/test_inv_1",
            },
          }),
        ),
      ),
    );
    render(<UsageLimitReachedCard />);
    const pay = await screen.findByRole("link", { name: "Pay $50 now" });
    expect(pay.getAttribute("href")).toBe(
      "https://invoice.stripe.com/i/acct_1/test_inv_1",
    );
    expect(screen.getByText("Your payment didn’t go through")).toBeDefined();
    for (const amount of screen.getAllByText(/\$50/)) {
      expect(amount.closest(".sentry-unmask, [data-sentry-unmask]")).toBeNull();
    }
    expect(screen.queryByRole("button", { name: /^Start/ })).toBeNull();
  });
  it("asks a cancel-pending trial to subscribe instead of starting Pro", async () => {
    setTrialUser();
    setup("TRIAL");
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json(
          trialResponse({
            allowance_used_percent: 100,
            usage_policy: "lifetime",
            cancel_at_period_end: true,
          }),
        ),
      ),
    );
    render(<UsageLimitReachedCard />);
    expect(
      await screen.findByRole("button", { name: "Subscribe now" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /^Start Pro/ })).toBeNull();
  });
});
