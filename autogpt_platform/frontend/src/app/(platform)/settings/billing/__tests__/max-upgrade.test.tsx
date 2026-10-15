import { setTrialUser } from "@/tests/integrations/trial-fixtures";
import { fireEvent, within } from "@testing-library/react";
import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { YourPlanCard } from "../components/SubscriptionTab/YourPlanCard/YourPlanCard";
import { maxUpgradePreview } from "./max-upgrade.fixture";

const subscription = {
  tier: "PRO",
  monthly_cost: 5000,
  tier_costs: { PRO: 5000, MAX: 32000 },
  tier_costs_yearly: { PRO: 51000, MAX: 326400 },
  tier_multipliers: { PRO: 2, MAX: 17 },
  billing_cycle: "monthly",
  has_active_stripe_subscription: true,
};

afterEach(() => setTrialUser(null));

beforeEach(() => {
  setTrialUser("billing-user");
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json(subscription),
    ),
    http.get("*/api/credits/manage", () =>
      HttpResponse.json({ url: "https://billing.stripe.com/test" }),
    ),
  );
});

async function openReview() {
  render(<YourPlanCard />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Upgrade to Max" }),
  );
  return within(await screen.findByRole("dialog", { name: /Upgrade to Max/ }));
}

describe("Max charge review", () => {
  it("discloses the actual debit, credit and tax, then confirms the reviewed timestamp", async () => {
    const preview = maxUpgradePreview();
    const requested = vi.fn();
    const charged = vi.fn();
    server.use(
      http.post("*/api/credits/subscription/preview", async ({ request }) => {
        requested(await request.json());
        return HttpResponse.json(preview);
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        charged(await request.json());
        return HttpResponse.json({ ...subscription, tier: "MAX" });
      }),
    );
    const dialog = await openReview();
    const pay = await dialog.findByRole("button", {
      name: "Pay $272.78 and upgrade",
    });
    expect(dialog.getByText("Due today")).toBeDefined();
    expect(dialog.getByText("$299.35")).toBeDefined();
    expect(dialog.getByText("-$46.77")).toBeDefined();
    expect(dialog.getByText("$20.20")).toBeDefined();
    expect(dialog.getByText(/8.5 times the usage of Pro/)).toBeDefined();
    expect(requested).toHaveBeenCalledWith({
      tier: "MAX",
      billing_cycle: "monthly",
    });
    expect(charged).not.toHaveBeenCalled();
    fireEvent.click(pay);
    await waitFor(() =>
      expect(charged).toHaveBeenCalledWith(
        expect.objectContaining({
          tier: "MAX",
          billing_cycle: "monthly",
          preview_id: preview.preview_id,
          proration_date: preview.proration_date,
        }),
      ),
    );
  });

  it("keeps the plan unchanged when the preview fails and provides an explicit retry", async () => {
    let attempts = 0;
    const charged = vi.fn();
    server.use(
      http.post("*/api/credits/subscription/preview", () =>
        ++attempts === 1
          ? new HttpResponse(null, { status: 502 })
          : HttpResponse.json(maxUpgradePreview()),
      ),
      http.post("*/api/credits/subscription", () => {
        charged();
        return HttpResponse.json(subscription);
      }),
    );
    const dialog = await openReview();
    await dialog.findByText(/We couldn’t load your charge/);
    expect(
      dialog
        .getByRole("button", { name: "Upgrade to Max" })
        .hasAttribute("disabled"),
    ).toBe(true);
    expect(charged).not.toHaveBeenCalled();
    fireEvent.click(dialog.getByRole("button", { name: "Retry review" }));
    await dialog.findByRole("button", { name: "Pay $272.78 and upgrade" });
    fireEvent.click(dialog.getByRole("button", { name: "Keep Pro" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(charged).not.toHaveBeenCalled();
  });

  it.each([
    { status: 409, message: /Your billing details changed/ },
    { status: 402, message: /Payment wasn’t completed/ },
  ])(
    "requires a fresh review after a $status response",
    async ({ status, message }) => {
      let quotes = 0;
      const charged = vi.fn();
      server.use(
        http.post("*/api/credits/subscription/preview", () =>
          HttpResponse.json(
            maxUpgradePreview(
              ++quotes === 1
                ? {}
                : {
                    preview_id: "b".repeat(32),
                    amount_due: 28000,
                    adjustment_amount: 722,
                  },
            ),
          ),
        ),
        http.post("*/api/credits/subscription", async ({ request }) => {
          charged(await request.json());
          return charged.mock.calls.length === 1
            ? HttpResponse.json({ detail: "Review again" }, { status })
            : HttpResponse.json({ ...subscription, tier: "MAX" });
        }),
      );
      const dialog = await openReview();
      fireEvent.click(
        await dialog.findByRole("button", { name: "Pay $272.78 and upgrade" }),
      );
      await dialog.findByText(message);
      expect(charged).toHaveBeenCalledTimes(1);
      expect(quotes).toBe(1);
      expect(dialog.queryByRole("button", { name: /Pay \$/ })).toBeNull();
      fireEvent.click(dialog.getByRole("button", { name: "Retry review" }));
      fireEvent.click(
        await dialog.findByRole("button", { name: "Pay $280.00 and upgrade" }),
      );
      await waitFor(() => expect(charged).toHaveBeenCalledTimes(2));
      expect(charged.mock.calls[1][0]).toMatchObject({
        preview_id: "b".repeat(32),
      });
    },
  );

  it("retries an uncertain payment with the same reviewed quote", async () => {
    const preview = maxUpgradePreview();
    const requested = vi.fn();
    const charged = vi.fn();
    server.use(
      http.post("*/api/credits/subscription/preview", () => {
        requested();
        return HttpResponse.json(preview);
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        charged(await request.json());
        return charged.mock.calls.length === 1
          ? HttpResponse.json(
              { detail: "Couldn’t confirm the payment result. Please retry." },
              { status: 502 },
            )
          : HttpResponse.json({ ...subscription, tier: "MAX" });
      }),
    );
    const dialog = await openReview();
    const pay = await dialog.findByRole("button", {
      name: "Pay $272.78 and upgrade",
    });
    fireEvent.click(pay);
    await waitFor(() => expect(charged).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(pay.hasAttribute("disabled")).toBe(false));
    expect(dialog.queryByRole("button", { name: "Retry review" })).toBeNull();
    fireEvent.click(pay);
    await waitFor(() => expect(charged).toHaveBeenCalledTimes(2));
    expect(requested).toHaveBeenCalledOnce();
    expect(charged.mock.calls[1][0]).toMatchObject({
      preview_id: preview.preview_id,
      proration_date: preview.proration_date,
    });
  });

  it("lets the user keep Pro while loading and cannot confirm before a quote arrives", async () => {
    server.use(
      http.post("*/api/credits/subscription/preview", async () => {
        await delay(100);
        return HttpResponse.json(maxUpgradePreview());
      }),
    );
    const dialog = await openReview();
    expect(
      dialog
        .getByRole("button", { name: "Upgrade to Max" })
        .hasAttribute("disabled"),
    ).toBe(true);
    fireEvent.click(dialog.getByRole("button", { name: "Keep Pro" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("uses yearly currency and tax-inclusive terms without adding tax twice", async () => {
    server.use(
      http.get("*/api/credits/subscription", () =>
        HttpResponse.json({ ...subscription, billing_cycle: "yearly" }),
      ),
      http.post("*/api/credits/subscription/preview", () =>
        HttpResponse.json(
          maxUpgradePreview({
            currency: "eur",
            billing_cycle: "yearly",
            recurring_amount: 326400,
            tax_included: true,
            recurring_tax_included: true,
          }),
        ),
      ),
    );
    const dialog = await openReview();
    await dialog.findByRole("button", { name: "Pay €272.78 and upgrade" });
    expect(dialog.getByText("Tax (included)")).toBeDefined();
    expect(
      dialog.getByText(
        /Max’s regular price is €3,264.00 including tax per year/,
      ),
    ).toBeDefined();
  });

  it("displays a zero-decimal currency in Stripe's minor units", async () => {
    server.use(
      http.post("*/api/credits/subscription/preview", () =>
        HttpResponse.json(maxUpgradePreview({ currency: "jpy" })),
      ),
    );
    const dialog = await openReview();
    await dialog.findByRole("button", { name: "Pay ¥27,278 and upgrade" });
    expect(dialog.getByText("¥29,935")).toBeDefined();
  });
});
