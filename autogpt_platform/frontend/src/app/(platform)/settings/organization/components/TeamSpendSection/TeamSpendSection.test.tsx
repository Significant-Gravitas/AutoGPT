import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { TeamSpendSection } from "./TeamSpendSection";

const SPEND_URL = "http://localhost:3000/api/proxy/api/orgs/org-1/spend";

const SPEND = {
  teams: [
    { team_id: null, team_name: null, total_spent: 1234, transaction_count: 5 },
    {
      team_id: "team-a",
      team_name: "Growth",
      total_spent: 4500,
      transaction_count: 12,
    },
  ],
};

describe("TeamSpendSection", () => {
  it("renders a spend row per team plus the org-home row", async () => {
    server.use(
      http.get(SPEND_URL, () => HttpResponse.json(SPEND, { status: 200 })),
    );

    render(<TeamSpendSection orgId="org-1" canManageBilling />);

    await waitFor(() =>
      expect(screen.getAllByTestId("team-spend-row")).toHaveLength(2),
    );
    // Org-home bucket (null team_id) is labelled "Organization".
    expect(screen.getByText("Organization")).toBeDefined();
    expect(screen.getByText("Growth")).toBeDefined();
    // Credits (cents) render as dollars, matching formatCredits.
    expect(screen.getByText("$12.34")).toBeDefined();
    expect(screen.getByText("$45.00")).toBeDefined();
  });

  it("renders nothing (and makes no request) for non-billing roles", () => {
    // No handler registered: if a request fired, onUnhandledRequest:"error"
    // would fail the test, proving the query stays disabled when gated out.
    render(<TeamSpendSection orgId="org-1" canManageBilling={false} />);

    expect(screen.queryByTestId("org-team-spend-section")).toBeNull();
  });
});
