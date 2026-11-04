import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";

import { AutomationCreditsTab } from "../components/AutomationCreditsTab/AutomationCreditsTab";
import { InvoicesCard } from "../components/SubscriptionTab/InvoicesCard/InvoicesCard";
import { PaymentMethodCard } from "../components/SubscriptionTab/PaymentMethodCard/PaymentMethodCard";

const MANAGED_MESSAGE =
  "Your organization's credits are managed by its owner and billing managers.";
const TOP_UP_REFUSED =
  "A team organization's credits are added by AutoGPT for now, so they can't be bought here.";

const TEAM_ORG = {
  id: "org-team",
  name: "Acme Inc",
  slug: "acme",
  avatarUrl: null,
  isPersonal: false,
  memberCount: 3,
};

const PERSONAL_ORG = {
  ...TEAM_ORG,
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  isPersonal: true,
  memberCount: 1,
};

function seedActiveOrg(org: typeof TEAM_ORG) {
  useOrgTeamStore.setState({
    activeOrgID: org.id,
    activeTeamID: null,
    orgs: [org],
    teams: [],
    isLoaded: true,
  });
}

// Records the X-Org-Id each credits route was called with ("none" when the
// request carried no organization).
function mockCreditsRoutes(balance: { credits: number } | "forbidden") {
  const calls: Record<string, string[]> = {};
  function record(route: string, request: Request) {
    (calls[route] ??= []).push(request.headers.get("X-Org-Id") ?? "none");
  }
  server.use(
    http.get("*/api/credits", ({ request }) => {
      record("balance", request);
      return balance === "forbidden"
        ? HttpResponse.json(
            { detail: "Missing org permission: MANAGE_BILLING" },
            { status: 403 },
          )
        : HttpResponse.json(balance);
    }),
    http.get("*/api/credits/transactions", ({ request }) => {
      record("history", request);
      return HttpResponse.json({
        transactions: [],
        next_transaction_time: null,
        next_cursor: null,
      });
    }),
    http.get("*/api/credits/auto-top-up", ({ request }) => {
      record("auto-top-up", request);
      return HttpResponse.json({ amount: 0, threshold: 0 });
    }),
    http.get("*/api/credits/invoices", ({ request }) => {
      record("invoices", request);
      return HttpResponse.json([]);
    }),
    http.get("*/api/credits/manage", ({ request }) => {
      record("manage", request);
      return HttpResponse.json({ url: "https://billing.stripe.com/p/test" });
    }),
  );
  return calls;
}

describe("Automation credits in a personal organization", () => {
  beforeEach(() => seedActiveOrg(PERSONAL_ORG));

  it("keeps the balance, top-up, auto-refill and history as they were", async () => {
    const calls = mockCreditsRoutes({ credits: 1234 });
    render(<AutomationCreditsTab />);

    expect(await screen.findByText("$12.34")).toBeDefined();
    expect(screen.getByRole("button", { name: "Add credits" })).toBeDefined();
    expect(screen.getByText("Auto-refill")).toBeDefined();
    expect(screen.queryByText(TOP_UP_REFUSED)).toBeNull();
    await waitFor(() => {
      expect(calls.history).toBeDefined();
    });
    // Requests still carry the active personal organization, as every request
    // already did.
    expect(new Set(calls.balance)).toEqual(new Set([PERSONAL_ORG.id]));
  });
});

describe("Automation credits in a team organization", () => {
  beforeEach(() => seedActiveOrg(TEAM_ORG));

  it("shows a billing manager the organization's balance and history, with no top-up", async () => {
    const calls = mockCreditsRoutes({ credits: 5000 });
    render(<AutomationCreditsTab />);

    expect(await screen.findByText("$50.00")).toBeDefined();
    expect(screen.getByText(TOP_UP_REFUSED)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Add credits" })).toBeNull();
    // A team organization's credits are added by AutoGPT: no auto-refill.
    expect(screen.queryByText("Auto-refill")).toBeNull();

    await waitFor(() => {
      expect(calls.history?.length).toBeGreaterThan(0);
    });
    expect(new Set(calls.balance)).toEqual(new Set([TEAM_ORG.id]));
    expect(new Set(calls.history)).toEqual(new Set([TEAM_ORG.id]));
    expect(calls["auto-top-up"]).toBeUndefined();
  });

  it("tells a plain member who manages the credits and loads nothing else", async () => {
    const calls = mockCreditsRoutes("forbidden");
    render(<AutomationCreditsTab />);

    expect(await screen.findByText(MANAGED_MESSAGE)).toBeDefined();
    expect(screen.queryByText("Add credits")).toBeNull();
    expect(screen.queryByText(TOP_UP_REFUSED)).toBeNull();
    expect(screen.queryByText("Auto-refill")).toBeNull();
    expect(
      screen.queryByText("We couldn't load your credit balance."),
    ).toBeNull();
    expect(calls.balance).toEqual([TEAM_ORG.id]);
    expect(calls.history).toBeUndefined();
    expect(calls["auto-top-up"]).toBeUndefined();
  });

  it("keeps a person's invoices and payment method on their own account", async () => {
    const calls = mockCreditsRoutes("forbidden");
    render(
      <>
        <PaymentMethodCard />
        <InvoicesCard />
      </>,
    );

    await waitFor(() => {
      expect(calls.invoices).toBeDefined();
      expect(calls.manage).toBeDefined();
    });
    // No organization on the request, so the backend answers for the
    // caller's own personal organization.
    expect(new Set(calls.invoices)).toEqual(new Set(["none"]));
    expect(new Set(calls.manage)).toEqual(new Set(["none"]));
  });
});
