import { server } from "@/mocks/mock-server";
import { getGetV2ListGrantsSharedWithMyTeamsQueryKey } from "@/app/api/__generated__/endpoints/grants/grants";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SharedWithTeamsSection } from "./SharedWithTeamsSection";

const TEAM_A = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

const RECEIVED_URL =
  "http://localhost:3000/api/proxy/api/orgs/org-1/grants/received";

const RECEIVED_GRANT = {
  id: "grant-1",
  agent_graph_id: "graph-9",
  agent_graph_version: 2,
  follow_latest: true,
  principal_id: "team-a",
  capability: "EXECUTE",
  credential_mode: "CONSUMER",
  graph_name: "Ops Copilot",
  graph_description: "Automates the ops runbook.",
  created_at: new Date("2026-07-01T00:00:00Z").toISOString(),
};

function seedTeams(teams: (typeof TEAM_A)[]) {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams,
    isLoaded: true,
  });
}

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS", "true");
  seedTeams([TEAM_A]);
});

afterEach(() => vi.unstubAllEnvs());

describe("SharedWithTeamsSection", () => {
  it("does not fetch received grants for an account with teams when disabled", () => {
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS", "false");
    const queryClient = new QueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <SharedWithTeamsSection />
      </QueryClientProvider>,
    );

    expect(screen.queryByTestId("shared-with-teams-section")).toBeNull();
    expect(
      queryClient.getQueryState(
        getGetV2ListGrantsSharedWithMyTeamsQueryKey("org-1"),
      )?.fetchStatus,
    ).toBe("idle");
  });

  it("hides already loaded shared agents when the flag turns off", async () => {
    server.use(
      http.get(RECEIVED_URL, () => HttpResponse.json([RECEIVED_GRANT])),
    );
    const view = render(<SharedWithTeamsSection />);
    await screen.findByText("Ops Copilot");

    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS", "false");
    view.rerender(<SharedWithTeamsSection />);

    expect(screen.queryByTestId("shared-with-teams-section")).toBeNull();
    expect(screen.queryByText("Ops Copilot")).toBeNull();
  });

  it("renders agents shared with the user's teams", async () => {
    server.use(
      http.get(RECEIVED_URL, () =>
        HttpResponse.json([RECEIVED_GRANT], { status: 200 }),
      ),
    );

    render(<SharedWithTeamsSection />);

    expect(
      await screen.findByTestId("shared-with-teams-section"),
    ).toBeDefined();
    expect(screen.getByText("Ops Copilot")).toBeDefined();
    expect(screen.getByText("Growth")).toBeDefined();
    expect(screen.getByText(/Can run · latest version/)).toBeDefined();
  });

  it("renders nothing (and makes no request) for solo users with no teams", () => {
    // No server.use handler: if a request fired, onUnhandledRequest:"error"
    // would fail the test, proving the query stays disabled for solo users.
    seedTeams([]);

    render(<SharedWithTeamsSection />);

    expect(screen.queryByTestId("shared-with-teams-section")).toBeNull();
  });

  it("renders nothing when no agents are shared with the teams", async () => {
    server.use(
      http.get(RECEIVED_URL, () => HttpResponse.json([], { status: 200 })),
    );

    render(<SharedWithTeamsSection />);

    await waitFor(() =>
      expect(screen.queryByTestId("shared-with-teams-section")).toBeNull(),
    );
  });
});
