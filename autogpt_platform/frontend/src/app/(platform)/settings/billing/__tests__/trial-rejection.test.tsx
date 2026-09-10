import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { TrialCheckoutConfirmation } from "@/components/organisms/TrialCard/TrialCheckoutConfirmation";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import SettingsBillingPage from "../page";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";

let searchParams = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useSearchParams: () => searchParams,
  usePathname: () => "/settings/billing",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));
beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

it.each([true, false])(
  "explains a used introductory offer on checkout return=%s without a retry dead end",
  async (isReturn) => {
    searchParams = new URLSearchParams(isReturn ? "trial=success" : "");
    const response = {
      ...trialResponse({ active: false, status: "canceled" }),
      rejection_reason: "intro_offer_already_used",
    };
    server.use(
      http.get("*/api/credits/trial", () => HttpResponse.json(response)),
      http.post("*/api/credits/trial/confirm", () =>
        HttpResponse.json(response),
      ),
    );
    render(
      <>
        <TrialCheckoutConfirmation />
        <TrialCard />
      </>,
    );
    await screen.findByText("This introductory offer has already been used");
    await waitFor(() =>
      expect(screen.queryByText(/Confirming your trial/)).toBeNull(),
    );
    expect(
      screen.getByText(/card or account has already redeemed/),
    ).toBeDefined();
    expect(screen.getByText(/choose a paid plan/i)).toBeDefined();
    expect(
      screen.queryByText(/Cancellation confirmed|review your card setup/i),
    ).toBeNull();
    expect(
      screen.queryByRole("button", {
        name: /try again|cancel trial|start.*trial/i,
      }),
    ).toBeNull();
  },
);

it("does not accuse a card of reuse when fingerprint verification fails", async () => {
  searchParams = new URLSearchParams("trial=success");
  const response = {
    ...trialResponse({ active: false, status: "canceled" }),
    rejection_reason: "card_verification_failed",
  };
  server.use(
    http.get("*/api/credits/trial", () => HttpResponse.json(response)),
    http.post("*/api/credits/trial/confirm", () => HttpResponse.json(response)),
  );
  render(
    <>
      <TrialCheckoutConfirmation />
      <TrialCard />
    </>,
  );
  await screen.findByText("We couldn’t verify your card for this trial");
  await waitFor(() =>
    expect(screen.queryByText(/Confirming your trial/)).toBeNull(),
  );
  expect(
    screen.queryByText(/already redeemed|Cancellation confirmed/),
  ).toBeNull();
  expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
});

it("keeps a paid-plan action and support available after rejection", async () => {
  searchParams = new URLSearchParams("trial=success");
  const response = trialResponse({
    active: false,
    status: "canceled",
    rejection_reason: "intro_offer_already_used",
  });
  server.use(
    http.get("*/api/credits/trial", () => HttpResponse.json(response)),
    http.post("*/api/credits/trial/confirm", () => HttpResponse.json(response)),
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({
        tier: "NO_TIER",
        monthly_cost: 0,
        has_active_stripe_subscription: false,
      }),
    ),
    http.get("*/api/credits/invoices", () => HttpResponse.json([])),
  );
  render(<SettingsBillingPage />);
  await screen.findByText("This introductory offer has already been used");
  expect(await screen.findByRole("button", { name: "Get Pro" })).toBeDefined();
  expect(
    screen.getByRole("link", { name: "Contact support" }).getAttribute("href"),
  ).toBe("https://discord.gg/autogpt");
});
