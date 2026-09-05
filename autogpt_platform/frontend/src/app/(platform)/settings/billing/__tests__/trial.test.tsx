import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";

const offer = {
  token: "a".repeat(64),
  version: "trial-v1",
  duration_days: 7,
  tier: "PRO",
  billing_cycle: "monthly",
  unit_amount: 2000,
  currency: "usd",
  onboarding_credit_amount: 300,
};

beforeEach(() => {
  useAuthStore.setState({
    user: { id: "trial-user", email: "trial@example.com", user_metadata: {} },
    isUserLoading: false,
    hasLoadedUser: true,
  });
});

afterEach(() => {
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

describe("trial billing", () => {
  it("does not describe a disabled pending checkout as an ended trial", async () => {
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json({
          eligible: false,
          offer,
          status: "checkout_pending",
        }),
      ),
    );
    const { container } = render(<TrialCard />);
    expect(container.querySelector(".animate-pulse")).not.toBeNull();
    await waitFor(() =>
      expect(container.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.queryByRole("region", { name: "AutoGPT trial" })).toBeNull();
  });

  it("does not advertise a zero-credit onboarding grant", async () => {
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json({
          eligible: true,
          offer: { ...offer, onboarding_credit_amount: 0 },
        }),
      ),
    );
    render(<TrialCard />);
    await screen.findByRole("button", { name: /start 7-day trial/i });
    expect(screen.queryByText(/0 one-time onboarding credits/i)).toBeNull();
  });

  it("shows the accepted duration, card requirement, and conversion price", async () => {
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json({ eligible: true, offer }),
      ),
    );
    render(<TrialCard />);
    expect(
      await screen.findByRole("button", { name: /start 7-day trial/i }),
    ).toBeDefined();
    expect(screen.getByText(/card required/i)).toBeDefined();
    expect(screen.getByText(/\$20\.00.*month/i)).toBeDefined();
    expect(screen.getByText(/one-time onboarding/i)).toBeDefined();
  });

  it("does not promise another grant to an existing onboarding recipient", async () => {
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json({
          eligible: true,
          offer,
          onboarding_credits_previously_received: true,
        }),
      ),
    );
    render(<TrialCard />);
    expect(
      await screen.findByText(
        /already received your one-time onboarding credits/i,
      ),
    ).toBeDefined();
  });
});
