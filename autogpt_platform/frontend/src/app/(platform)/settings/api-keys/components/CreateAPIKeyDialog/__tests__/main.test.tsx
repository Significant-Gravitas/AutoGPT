import { getPostV1CreateNewApiKeyMockHandler } from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";
import type { CreateAPIKeyResponse } from "@/app/api/__generated__/models/createAPIKeyResponse";
import {
  CreateSurface,
  setLastUsedTeam,
} from "@/components/contextual/TeamPicker/helpers";
import { server } from "@/mocks/mock-server";
import { ORG_HEADER_NAME, TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CreateAPIKeyDialog } from "../CreateAPIKeyDialog";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ user: { id: "user-owner" } }),
}));

const TEAM = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-team",
};

const CREATED: CreateAPIKeyResponse = {
  plain_text_key: "sk-test-123",
  api_key: {
    id: "key-1",
    name: "CI key",
    head: "sk-test",
    tail: "123",
    status: "ACTIVE",
    permissions: ["EXECUTE_GRAPH"],
    scopes: [],
    user_id: "user-owner",
    created_at: new Date("2026-01-01T00:00:00Z"),
  } as unknown as CreateAPIKeyResponse["api_key"],
};

beforeEach(() => {
  window.localStorage.clear();
  useOrgTeamStore.setState({
    activeOrgID: "org-team",
    activeTeamID: TEAM.id,
    orgs: [],
    teams: [TEAM],
    isLoaded: true,
  });
});

afterEach(() => vi.unstubAllEnvs());

function captureCreate() {
  const sent: { org: string | null; team: string | null }[] = [];
  server.use(
    getPostV1CreateNewApiKeyMockHandler((info) => {
      sent.push({
        org: info.request.headers.get(ORG_HEADER_NAME),
        team: info.request.headers.get(TEAM_HEADER_NAME),
      });
      return CREATED;
    }),
  );
  return sent;
}

async function createKey() {
  await userEvent.type(screen.getByLabelText("Name"), "CI key");
  fireEvent.click(screen.getAllByRole("checkbox")[0]);
  await userEvent.click(screen.getByRole("button", { name: "Create Key" }));
}

describe("CreateAPIKeyDialog scope", () => {
  it.each([true, false])(
    "creates the key in your personal organization with rollout=%s",
    async (enabled) => {
      vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS", String(enabled));
      setLastUsedTeam("org-team", CreateSurface.ApiKey, TEAM.id);
      const sent = captureCreate();

      render(<CreateAPIKeyDialog open onOpenChange={() => {}} />);
      await createKey();

      // No scope headers: the server picks your personal organization and its
      // default team, whichever organization is active here.
      await waitFor(() => expect(sent).toHaveLength(1));
      expect(sent[0]).toEqual({ org: null, team: null });
    },
  );

  it("offers no team to restrict the key to", async () => {
    vi.stubEnv("NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS", "true");

    render(<CreateAPIKeyDialog open onOpenChange={() => {}} />);

    expect(await screen.findByLabelText("Name")).toBeDefined();
    expect(
      screen.queryByRole("combobox", { name: "Restrict to team" }),
    ).toBeNull();
  });
});
