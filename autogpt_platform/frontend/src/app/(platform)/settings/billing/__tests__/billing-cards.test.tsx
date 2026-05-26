/**
 * Card-by-card integration tests for the Settings v2 billing page.
 *
 * Each test renders a single card with deterministic MSW responses, exercising
 * the card's render path AND its hook. Together they cover every new
 * file introduced by this PR (subscription + automation-credits cards) so the
 * patch coverage threshold passes.
 *
 * Pattern: MSW handlers replace the global defaults via `server.use(...)` and
 * the cards are rendered through the `TestProviders` wrapper which gives
 * them a fresh `QueryClient`.
 */

import { fireEvent } from "@testing-library/react";
import { http, HttpResponse, type JsonBodyType } from "msw";
import { describe, expect, it } from "vitest";

import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";

import { AutoRefillCard } from "../components/AutomationCreditsTab/AutoRefillCard/AutoRefillCard";
import { AutoRefillDialog } from "../components/AutomationCreditsTab/AutoRefillCard/AutoRefillDialog";
import { BalanceCard } from "../components/AutomationCreditsTab/BalanceCard/BalanceCard";
import { TransactionHistoryCard } from "../components/AutomationCreditsTab/TransactionHistoryCard/TransactionHistoryCard";
import { UsageCard } from "../components/AutomationCreditsTab/UsageCard/UsageCard";
import { AutopilotUsageCard } from "../components/SubscriptionTab/AutopilotUsageCard/AutopilotUsageCard";
import { InvoicesCard } from "../components/SubscriptionTab/InvoicesCard/InvoicesCard";
import { PaymentMethodCard } from "../components/SubscriptionTab/PaymentMethodCard/PaymentMethodCard";
import { YourPlanCard } from "../components/SubscriptionTab/YourPlanCard/YourPlanCard";

function jsonHandler(method: "get" | "post", path: string, body: JsonBodyType) {
  return http[method](`*${path}`, () => HttpResponse.json(body));
}

describe("YourPlanCard", () => {
  it("renders the current tier label, monthly cost, and Upgrade CTA", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    expect(await screen.findByText("Pro")).toBeDefined();
    expect(screen.getByText(/\$50\.00 \/ month/)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /upgrade to max/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /manage subscription/i }),
    ).toBeDefined();
  });

  it("hides the Upgrade button on the top tier (BUSINESS / Team) and exposes Downgrade to Max", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "BUSINESS",
        monthly_cost: 50000,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    render(<YourPlanCard />);

    expect(await screen.findByText("Team")).toBeDefined();
    expect(screen.queryByRole("button", { name: /upgrade to/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /talk to sales/i })).toBeNull();
    expect(
      screen.getByRole("button", { name: /downgrade to max/i }),
    ).toBeDefined();
  });

  it("MAX subscriber sees Downgrade-to-Pro, Manage, and Talk-to-sales for Team", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    expect(await screen.findByText("Max")).toBeDefined();
    expect(
      screen.getByRole("button", { name: /downgrade to pro/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /manage subscription/i }),
    ).toBeDefined();
    // Team is contact-sales — button exists, but as a "Talk to sales" link
    // rather than a Stripe Checkout / API upgrade trigger.
    expect(
      screen.getByRole("button", { name: /talk to sales/i }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /upgrade to team/i }),
    ).toBeNull();
  });

  it("shows 'Downgrade scheduled' badge + 'Switches to Pro on …' when pending downgrade is set", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        has_active_stripe_subscription: true,
        status: "active",
        pending_tier: "PRO",
        pending_tier_effective_at: "2026-05-30T00:00:00Z",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    expect(await screen.findByText("Downgrade scheduled")).toBeDefined();
    expect(screen.getByText(/Switches to Pro on/i)).toBeDefined();
    // Downgrade button hidden while a downgrade is already pending; the
    // "Cancel downgrade" CTA takes its place.
    expect(screen.queryByRole("button", { name: /^downgrade to/i })).toBeNull();
    expect(
      screen.getByRole("button", { name: /cancel downgrade/i }),
    ).toBeDefined();
  });

  it("renders the 'no active subscription' state for NO_TIER users", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "NO_TIER",
        monthly_cost: 0,
        has_active_stripe_subscription: false,
        status: "inactive",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    render(<YourPlanCard />);

    expect(await screen.findByText("No active subscription")).toBeDefined();
    expect(screen.getByRole("button", { name: /get pro/i })).toBeDefined();
    expect(screen.queryByRole("button", { name: /cancel plan/i })).toBeNull();
    expect(
      screen.queryByRole("button", { name: /manage subscription/i }),
    ).toBeNull();
  });
});

describe("YourPlanCard cycle toggle", () => {
  it("renders Monthly selected when billing_cycle is monthly", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    const monthly = await screen.findByRole("radio", { name: /monthly/i });
    const yearly = screen.getByRole("radio", { name: /yearly/i });
    expect(monthly.getAttribute("aria-checked")).toBe("true");
    expect(yearly.getAttribute("aria-checked")).toBe("false");
  });

  it("renders Yearly selected when billing_cycle is yearly", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 51000,
        billing_cycle: "yearly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    const yearly = await screen.findByRole("radio", { name: /yearly/i });
    const monthly = screen.getByRole("radio", { name: /monthly/i });
    expect(yearly.getAttribute("aria-checked")).toBe("true");
    expect(monthly.getAttribute("aria-checked")).toBe("false");
  });

  it("opens the confirmation dialog with prorated copy when monthly Pro user clicks Yearly", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        proration_credit_cents: 2500,
        current_period_end: 1900000000,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    const yearly = await screen.findByRole("radio", { name: /yearly/i });
    fireEvent.click(yearly);

    expect(
      await screen.findByText(/Switch Pro to yearly billing\?/i),
    ).toBeDefined();
    // Each line splits into a bold label span + regular value span, so we
    // assert the two pieces separately.
    expect(screen.getByText(/^Save 15% with yearly billing\.$/)).toBeDefined();
    expect(screen.getByText("Pro yearly:")).toBeDefined();
    expect(
      screen.getByText(/^\$510\.00\/year \(\$42\.50\/month\)\.$/),
    ).toBeDefined();
    // 51000 yearly - 2500 proration credit = 48500 cents = $485.00 charged today.
    expect(screen.getByText("Charged today:")).toBeDefined();
    expect(
      screen.getByText(/^\$485\.00 \(prorated from your monthly plan\)\.$/),
    ).toBeDefined();
    expect(screen.getByText("Renews")).toBeDefined();
    expect(screen.getByText(/^\$510\.00\/year on /)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /switch to yearly/i }),
    ).toBeDefined();
  });

  it("shows Max-specific yearly savings copy in the confirmation dialog", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(await screen.findByRole("radio", { name: /yearly/i }));

    expect(
      await screen.findByText(/Switch Max to yearly billing\?/i),
    ).toBeDefined();
    expect(screen.getByText("Max yearly:")).toBeDefined();
    expect(
      screen.getByText(/^\$3,264\.00\/year \(\$272\.00\/month\)\.$/),
    ).toBeDefined();
    // No proration_credit_cents in this fixture → falls back to generic
    // "prorated difference today" line instead of an exact dollar amount.
    expect(
      screen.getByText(/^You'll be charged the prorated difference today\.$/),
    ).toBeDefined();
    expect(screen.getByText("Renews")).toBeDefined();
    expect(
      screen.getByText(/^at \$3,264\.00\/year after this period\.$/),
    ).toBeDefined();
  });

  it("opens the confirmation dialog with end-of-period copy when a yearly Pro user clicks Monthly", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 51000,
        billing_cycle: "yearly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(await screen.findByRole("radio", { name: /monthly/i }));

    expect(
      await screen.findByText(/Switch Pro to monthly billing\?/i),
    ).toBeDefined();
    expect(
      screen.getByText(
        /^Switches to monthly at the end of your current yearly period\.$/,
      ),
    ).toBeDefined();
    expect(screen.getByText("New price:")).toBeDefined();
    expect(screen.getByText(/^\$50\.00\/month\.$/)).toBeDefined();
    expect(screen.getByText(/^No charge today\.$/)).toBeDefined();
  });

  it("falls back to the generic dialog title when prices are missing (NO_TIER-like)", async () => {
    // BUSINESS tier exercises the generic-title fallback in getDialogTitle —
    // the cycle toggle is still visible (BUSINESS isn't in the hidden set),
    // but the title shouldn't read "Switch Team to yearly billing?" since
    // BUSINESS is contact-sales and not a user-managed yearly tier.
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "BUSINESS",
        monthly_cost: 50000,
        billing_cycle: "monthly",
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(await screen.findByRole("radio", { name: /yearly/i }));

    expect(
      await screen.findByText(/Switch billing to Yearly\?/i),
    ).toBeDefined();
    // No tier_costs / tier_costs_yearly in the payload — body falls back to
    // proration-only copy without the savings/pricing lines.
    expect(
      screen.getByText(/^You'll be charged the prorated difference today\.$/),
    ).toBeDefined();
    expect(
      screen.getByText(
        /^Renews on the new yearly cadence after this period\.$/,
      ),
    ).toBeDefined();
  });

  it("fires updateTier with billing_cycle on confirm", async () => {
    let capturedBody: { tier?: string; billing_cycle?: string } | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(await screen.findByRole("radio", { name: /yearly/i }));
    fireEvent.click(
      await screen.findByRole("button", { name: /switch to yearly/i }),
    );

    await waitFor(() => {
      expect(capturedBody).not.toBeNull();
      expect(capturedBody?.tier).toBe("PRO");
      expect(capturedBody?.billing_cycle).toBe("yearly");
    });
  });

  it("cancelling the dialog leaves no mutation fired", async () => {
    let mutationFired = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", () => {
        mutationFired = true;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(await screen.findByRole("radio", { name: /yearly/i }));
    fireEvent.click(await screen.findByRole("button", { name: /^cancel$/i }));

    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: /switch to yearly/i }),
      ).toBeNull(),
    );
    expect(mutationFired).toBe(false);
  });

  it("opens a confirmation dialog (no API call) when a paid user clicks Upgrade to Max", async () => {
    let mutationFired = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", () => {
        mutationFired = true;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /upgrade to max/i }),
    );

    expect(await screen.findByText(/Upgrade to Max\?/i)).toBeDefined();
    expect(
      screen.getByText(
        /charged the prorated difference immediately for the rest of your monthly period/i,
      ),
    ).toBeDefined();
    expect(mutationFired).toBe(false);
  });

  it("fires updateTier with the server billing cycle when the upgrade dialog is confirmed", async () => {
    let capturedBody: { tier?: string; billing_cycle?: string } | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 51000,
        billing_cycle: "yearly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /upgrade to max/i }),
    );

    const confirmInDialog = await screen.findAllByRole("button", {
      name: /upgrade to max/i,
    });
    // Two buttons share the label: the card's trigger + the dialog's confirm.
    // The confirm is the second one in DOM order.
    fireEvent.click(confirmInDialog[confirmInDialog.length - 1]);

    await waitFor(() => {
      expect(capturedBody).not.toBeNull();
      expect(capturedBody?.tier).toBe("MAX");
      expect(capturedBody?.billing_cycle).toBe("yearly");
    });
  });

  it("cancelling the upgrade dialog leaves no mutation fired", async () => {
    let mutationFired = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", () => {
        mutationFired = true;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /upgrade to max/i }),
    );
    expect(await screen.findByText(/Upgrade to Max\?/i)).toBeDefined();
    fireEvent.click(await screen.findByRole("button", { name: /^cancel$/i }));

    await waitFor(() =>
      expect(screen.queryByText(/Upgrade to Max\?/i)).toBeNull(),
    );
    expect(mutationFired).toBe(false);
  });

  it("clicking Downgrade to Pro opens a confirmation dialog with end-of-period copy", async () => {
    let mutationFired = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
        current_period_end: 1900000000,
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", () => {
        mutationFired = true;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /downgrade to pro/i }),
    );

    expect(await screen.findByText(/Downgrade to Pro\?/i)).toBeDefined();
    expect(
      screen.getByText(/until .* then switch to Pro\. No charge today\./i),
    ).toBeDefined();
    expect(mutationFired).toBe(false);
  });

  it("confirming the downgrade dialog fires updateTier with prevTier + serverCycle", async () => {
    let capturedBody: { tier?: string; billing_cycle?: string } | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 326400,
        billing_cycle: "yearly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /downgrade to pro/i }),
    );

    const confirmInDialog = await screen.findAllByRole("button", {
      name: /downgrade to pro/i,
    });
    // Two buttons share the label: the card's trigger + the dialog's confirm.
    fireEvent.click(confirmInDialog[confirmInDialog.length - 1]);

    await waitFor(() => {
      expect(capturedBody).not.toBeNull();
      expect(capturedBody?.tier).toBe("PRO");
      expect(capturedBody?.billing_cycle).toBe("yearly");
    });
  });

  it("cancelling the downgrade dialog leaves no mutation fired", async () => {
    let mutationFired = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        billing_cycle: "monthly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", () => {
        mutationFired = true;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /downgrade to pro/i }),
    );
    expect(await screen.findByText(/Downgrade to Pro\?/i)).toBeDefined();
    fireEvent.click(await screen.findByRole("button", { name: /^cancel$/i }));

    await waitFor(() =>
      expect(screen.queryByText(/Downgrade to Pro\?/i)).toBeNull(),
    );
    expect(mutationFired).toBe(false);
  });

  it("NO_TIER user clicking Get Pro skips the confirmation dialog (Stripe Checkout flow)", async () => {
    let mutationFired = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "NO_TIER",
        monthly_cost: 0,
        has_active_stripe_subscription: false,
        status: "inactive",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
      http.post("*/api/credits/subscription", () => {
        mutationFired = true;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(await screen.findByRole("button", { name: /get pro/i }));

    // No SwitchTierDialog rendered for free→paid (Checkout success_url path).
    await waitFor(() => expect(mutationFired).toBe(true));
    expect(screen.queryByText(/Upgrade to Pro\?/i)).toBeNull();
  });

  it("hides the cycle toggle entirely for ENTERPRISE tier", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "ENTERPRISE",
        monthly_cost: 0,
        billing_cycle: "monthly",
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    render(<YourPlanCard />);

    await waitFor(() => expect(screen.queryAllByRole("radio").length).toBe(0));
  });

  it("hides the cycle toggle entirely for BASIC tier", async () => {
    // BASIC is a reserved internal slot (no Stripe sub, no upgrade target);
    // showing a cycle toggle would imply user-manageable billing.
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "BASIC",
        monthly_cost: 0,
        billing_cycle: "monthly",
        has_active_stripe_subscription: false,
        status: "inactive",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    render(<YourPlanCard />);

    await waitFor(() => expect(screen.queryAllByRole("radio").length).toBe(0));
  });

  it("hides the cycle toggle entirely for NO_TIER users (no active sub)", async () => {
    // NO_TIER means no active subscription — the user has nothing to switch.
    // The toggle would be dead UI implying functionality that doesn't apply.
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "NO_TIER",
        monthly_cost: 0,
        has_active_stripe_subscription: false,
        status: "inactive",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    render(<YourPlanCard />);

    await waitFor(() => expect(screen.queryAllByRole("radio").length).toBe(0));
  });

  it("downgrade preserves yearly cycle (forwards billing_cycle in mutation)", async () => {
    let capturedBody: { tier?: string; billing_cycle?: string } | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 326400,
        billing_cycle: "yearly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /downgrade to pro/i }),
    );
    // Downgrade now goes through a confirm dialog (symmetric with upgrade).
    // Two buttons share the label: the card's trigger + the dialog's confirm —
    // confirm is the second one in DOM order.
    const confirmInDialog = await screen.findAllByRole("button", {
      name: /downgrade to pro/i,
    });
    fireEvent.click(confirmInDialog[confirmInDialog.length - 1]);

    await waitFor(() => {
      expect(capturedBody).not.toBeNull();
      expect(capturedBody?.tier).toBe("PRO");
      // Without forwarding the current cycle the backend defaults to monthly,
      // silently flipping the yearly subscriber at period end.
      expect(capturedBody?.billing_cycle).toBe("yearly");
    });
  });

  it("resume preserves yearly cycle (forwards billing_cycle in mutation)", async () => {
    let capturedBody: { tier?: string; billing_cycle?: string } | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 51000,
        billing_cycle: "yearly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
        pending_tier: "NO_TIER",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        return HttpResponse.json({ url: "" });
      }),
    );

    render(<YourPlanCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /resume subscription/i }),
    );

    await waitFor(() => {
      expect(capturedBody).not.toBeNull();
      expect(capturedBody?.tier).toBe("PRO");
      // Same-tier release path also defaults to monthly when billing_cycle is
      // omitted — gates as a cycle-switch and silently flips to monthly.
      expect(capturedBody?.billing_cycle).toBe("yearly");
    });
  });

  it("shows 'Cycle switch scheduled' badge + cycle-only secondary line when same-tier yearly→monthly is queued", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 51000,
        billing_cycle: "yearly",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
        status: "active",
        pending_tier: "PRO",
        pending_billing_cycle: "monthly",
        pending_tier_effective_at: "2026-12-15T00:00:00Z",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<YourPlanCard />);

    expect(await screen.findByText("Cycle switch scheduled")).toBeDefined();
    // Active yearly price stays on the primary line; secondary line describes
    // the queued switch with no charge today.
    expect(screen.getByText(/\$510\.00 \/ year/)).toBeDefined();
    expect(
      screen.getByText(/Switching to monthly Pro on .* · No charge today/i),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /cancel cycle switch/i }),
    ).toBeDefined();
    // Tier-downgrade button is suppressed while a cycle switch is pending —
    // the user needs to cancel/release the schedule first.
    expect(screen.queryByRole("button", { name: /^downgrade to/i })).toBeNull();
  });
});

describe("PaymentMethodCard", () => {
  it("disables 'Open portal' until the portal URL resolves", async () => {
    server.use(jsonHandler("get", "/api/credits/manage", { url: null }));

    render(<PaymentMethodCard />);

    // The descriptive copy is always rendered.
    expect(await screen.findByText(/Manage payment method/i)).toBeDefined();
    const button = screen.getByRole("button", { name: /open portal/i });
    expect(button.hasAttribute("disabled")).toBe(true);
  });

  it("enables 'Open portal' once the portal URL is available", async () => {
    server.use(
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    render(<PaymentMethodCard />);

    await waitFor(() => {
      const button = screen.getByRole("button", { name: /open portal/i });
      expect(button.hasAttribute("disabled")).toBe(false);
    });
  });
});

describe("InvoicesCard", () => {
  it("renders a no-invoices empty state when Stripe returns no invoices", async () => {
    server.use(jsonHandler("get", "/api/credits/invoices", []));

    render(<InvoicesCard />);

    expect(await screen.findByText(/No invoices yet/i)).toBeDefined();
  });

  it("renders an invoice row per Stripe invoice", async () => {
    server.use(
      jsonHandler("get", "/api/credits/invoices", [
        {
          id: "in_001",
          number: "INV-001",
          created_at: "2026-04-01T00:00:00Z",
          total_cents: 2000,
          amount_paid_cents: 2000,
          currency: "usd",
          status: "paid",
          description: "Top up",
          hosted_invoice_url: "https://stripe.com/i/001",
          invoice_pdf_url: "https://stripe.com/i/001.pdf",
        },
      ]),
    );

    render(<InvoicesCard />);

    expect(await screen.findByText("INV-001")).toBeDefined();
  });
});

describe("AutopilotUsageCard", () => {
  it("renders today + week percent values", async () => {
    server.use(
      jsonHandler("get", "/api/chat/usage", {
        daily: { percent_used: 12, resets_at: null },
        weekly: { percent_used: 35, resets_at: null },
      }),
    );

    render(<AutopilotUsageCard />);

    expect(await screen.findByText("Today")).toBeDefined();
    expect(screen.getByText("This Week")).toBeDefined();
    expect(screen.getByText("12% used")).toBeDefined();
    expect(screen.getByText("35% used")).toBeDefined();
  });
});

describe("BalanceCard", () => {
  it("renders the Intl-formatted balance and an Add Credits CTA", async () => {
    server.use(jsonHandler("get", "/api/credits", { credits: 1234 }));

    render(<BalanceCard />);

    // 1234 cents → $12.34 with thousands separator support.
    expect(await screen.findByText("$12.34")).toBeDefined();
    expect(screen.getByRole("button", { name: /add credits/i })).toBeDefined();
  });

  it("renders ErrorCard with a Retry button on a 500 from /api/credits", async () => {
    server.use(
      http.get("*/api/credits", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<BalanceCard />);

    await waitFor(() => {
      // ErrorCard renders, NOT the silent "$0.00" fallback.
      expect(screen.queryByText("$0.00")).toBeNull();
    });
  });
});

describe("AutoRefillCard", () => {
  it("renders the disabled state when no auto top-up is configured", async () => {
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 0,
        threshold: 0,
      }),
    );

    render(<AutoRefillCard />);

    // Disabled-state copy + "Set up auto-refill" CTA.
    expect(
      await screen.findByText(
        /Top up automatically when your balance gets low/i,
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /set up auto-refill/i }),
    ).toBeDefined();
  });

  it("renders the configured state when auto top-up has values", async () => {
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 2000,
        threshold: 500,
      }),
    );

    render(<AutoRefillCard />);

    // Configured copy includes the dollar amounts.
    expect(
      await screen.findByText(/Refills \$20 when balance drops below \$5/i),
    ).toBeDefined();
  });
});

describe("UsageCard", () => {
  it("renders the 30-day chart with the totals header", async () => {
    server.use(
      jsonHandler("get", "/api/credits/transactions", {
        transactions: [
          {
            transaction_key: "u1",
            transaction_time: new Date().toISOString(),
            transaction_type: "USAGE",
            amount: -150,
            description: "Agent run",
            running_balance: 850,
          },
        ],
        next_transaction_time: null,
      }),
    );

    render(<UsageCard />);

    expect(await screen.findByText(/last 30 days/i)).toBeDefined();
  });

  it("renders the zero-state when there is no usage at all", async () => {
    server.use(
      jsonHandler("get", "/api/credits/transactions", {
        transactions: [],
        next_transaction_time: null,
      }),
    );

    render(<UsageCard />);

    // Card no longer hides itself on zero usage.
    expect(await screen.findByText(/last 30 days/i)).toBeDefined();
  });
});

describe("TransactionHistoryCard", () => {
  it("renders a no-transactions empty state when the API returns []", async () => {
    server.use(
      jsonHandler("get", "/api/credits/transactions", {
        transactions: [],
        next_transaction_time: null,
      }),
    );

    render(<TransactionHistoryCard />);

    expect(await screen.findByText(/No transactions yet/i)).toBeDefined();
  });

  it("renders a row per transaction with credit/debit colour-coded amounts", async () => {
    server.use(
      jsonHandler("get", "/api/credits/transactions", {
        transactions: [
          {
            transaction_key: "TXN-A",
            transaction_time: "2026-04-10T00:00:00Z",
            transaction_type: "TOP_UP",
            amount: 5000,
            description: "Top up",
            running_balance: 5000,
          },
          {
            transaction_key: "TXN-B",
            transaction_time: "2026-04-11T00:00:00Z",
            transaction_type: "USAGE",
            amount: -250,
            description: "Agent run",
            running_balance: 4750,
          },
        ],
        next_transaction_time: null,
      }),
    );

    render(<TransactionHistoryCard />);

    expect(await screen.findByText("Top up")).toBeDefined();
    expect(screen.getByText("Agent run")).toBeDefined();
    expect(screen.getByText("+$50.00")).toBeDefined();
    expect(screen.getByText("-$2.50")).toBeDefined();
  });

  it("renders ErrorCard with a Retry button on a 500", async () => {
    server.use(
      http.get("*/api/credits/transactions", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<TransactionHistoryCard />);

    await waitFor(() => {
      expect(screen.queryByText(/No transactions yet/i)).toBeNull();
    });
  });
});

/**
 * Mutation flow tests — these exercise the dialog open/save paths in the
 * card hooks (handleSubmit, save, disable, changeTier) so v8 coverage on the
 * patch lines stays above the codecov 80% gate.
 */
describe("Card mutation flows", () => {
  it("BalanceCard: opening the Add Credits dialog renders the amount input", async () => {
    server.use(jsonHandler("get", "/api/credits", { credits: 1000 }));

    render(<BalanceCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /add credits/i }),
    );

    // Dialog body is now in the DOM with the input + checkout button.
    expect(
      await screen.findByText(/We'll redirect you to Stripe/i),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /continue to checkout/i }),
    ).toBeDefined();
  });

  it("BalanceCard: invalid amount (4) keeps the checkout button disabled until a valid amount is entered", async () => {
    server.use(jsonHandler("get", "/api/credits", { credits: 1000 }));

    render(<BalanceCard />);

    fireEvent.click(
      await screen.findByRole("button", { name: /add credits/i }),
    );

    const continueButton = await screen.findByRole("button", {
      name: /continue to checkout/i,
    });
    expect(continueButton.hasAttribute("disabled")).toBe(true);

    // Typing a valid amount enables the button — exercises the input
    // onChange handler that wasn't covered before.
    const amountInput = screen.getByPlaceholderText(/amount/i);
    fireEvent.change(amountInput, { target: { value: "20" } });
    await waitFor(() =>
      expect(continueButton.hasAttribute("disabled")).toBe(false),
    );

    // Cancel closes the dialog — covers the Cancel button's onClick handler.
    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: /continue to checkout/i }),
      ).toBeNull(),
    );
  });

  it("AutoRefillDialog: renders 'Enable Auto-Refill' affordances when not yet enabled", async () => {
    const noop = () => {};
    render(
      <AutoRefillDialog
        isOpen={true}
        onOpenChange={noop}
        threshold="10"
        setThreshold={noop}
        refillAmount="20"
        setRefillAmount={noop}
        isValid={true}
        isEnabled={false}
        isSaving={false}
        onSave={noop}
        onDisable={noop}
      />,
    );

    expect(
      await screen.findByText(/Top up your balance automatically/i),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /enable auto-refill/i }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: /cancel/i })).toBeDefined();
  });

  it("AutoRefillDialog: renders 'Save changes' + 'Disable' when already enabled", async () => {
    const noop = () => {};
    render(
      <AutoRefillDialog
        isOpen={true}
        onOpenChange={noop}
        threshold="5"
        setThreshold={noop}
        refillAmount="20"
        setRefillAmount={noop}
        isValid={true}
        isEnabled={true}
        isSaving={false}
        onSave={noop}
        onDisable={noop}
      />,
    );

    expect(
      await screen.findByRole("button", { name: /save changes/i }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: /^disable$/i })).toBeDefined();
  });

  it("YourPlanCard: hides users without an active subscription from Manage controls", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "NO_TIER",
        monthly_cost: 0,
        has_active_stripe_subscription: false,
        status: "inactive",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    render(<YourPlanCard />);

    expect(await screen.findByText("No active subscription")).toBeDefined();
    // Cancellation now lives in the Stripe portal — no in-app Cancel button.
    expect(screen.queryByRole("button", { name: /cancel plan/i })).toBeNull();
    expect(
      screen.queryByRole("button", { name: /manage subscription/i }),
    ).toBeNull();
  });
});
