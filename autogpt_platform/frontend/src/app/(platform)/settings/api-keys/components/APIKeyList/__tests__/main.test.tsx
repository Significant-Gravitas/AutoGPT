import { getGetV1ListUserApiKeysMockHandler } from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";
import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { APIKeyList } from "../APIKeyList";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ user: { id: "user-owner" } }),
}));

const TEAM_A = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

function apiKey(overrides: Partial<APIKeyInfo>): APIKeyInfo {
  return {
    id: "key-x",
    name: "Key",
    head: "sk-abcd",
    tail: "1234",
    status: "ACTIVE",
    permissions: ["EXECUTE_GRAPH"],
    scopes: [],
    user_id: "user-owner",
    created_at: new Date("2026-01-01T00:00:00Z"),
    ...overrides,
  } as unknown as APIKeyInfo;
}

beforeEach(() => {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams: [TEAM_A],
    isLoaded: true,
  });
});

describe("APIKeyList team badges", () => {
  it("shows the team name badge only on team-restricted keys", async () => {
    server.use(
      getGetV1ListUserApiKeysMockHandler([
        apiKey({
          id: "key-team",
          name: "Team key",
          team_id_restriction: "team-a",
        }),
        apiKey({ id: "key-org", name: "Org key", team_id_restriction: null }),
      ]),
    );

    render(<APIKeyList />);

    // The restricted key surfaces the team name badge.
    expect(await screen.findByText("Team key")).toBeTruthy();
    expect(await screen.findByText("Growth")).toBeTruthy();

    // Exactly one badge — the org-home key gets none.
    await waitFor(() => {
      expect(screen.getAllByText("Growth")).toHaveLength(1);
    });
    expect(screen.getByText("Org key")).toBeTruthy();
  });
});
