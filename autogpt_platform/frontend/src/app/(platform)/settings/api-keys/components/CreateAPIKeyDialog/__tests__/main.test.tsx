import { getPostV1CreateNewApiKeyMockHandler } from "@/app/api/__generated__/endpoints/api-keys/api-keys.msw";
import type { CreateAPIKeyResponse } from "@/app/api/__generated__/models/createAPIKeyResponse";
import { server } from "@/mocks/mock-server";
import { TEAM_HEADER_NAME } from "@/services/org-team/headers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
  orgId: "org-1",
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
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams: [TEAM],
    isLoaded: true,
  });
});

describe("CreateAPIKeyDialog team restriction", () => {
  it("sends the picked team as the X-Team-Id header on create", async () => {
    let sentTeamHeader: string | null = null;
    const createSpy = vi.fn();
    server.use(
      getPostV1CreateNewApiKeyMockHandler((info) => {
        createSpy();
        sentTeamHeader = info.request.headers.get(TEAM_HEADER_NAME);
        return CREATED;
      }),
    );

    render(<CreateAPIKeyDialog open onOpenChange={() => {}} />);

    await userEvent.type(screen.getByLabelText("Name"), "CI key");
    fireEvent.click(screen.getAllByRole("checkbox")[0]);

    // Pick the team (defaults to Organization / org-home otherwise).
    fireEvent.click(screen.getByRole("combobox", { name: "Restrict to team" }));
    fireEvent.click(await screen.findByRole("option", { name: "Growth" }));

    await userEvent.click(screen.getByRole("button", { name: "Create Key" }));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(sentTeamHeader).toBe(TEAM.id);
  });

  it("omits the team header when Organization (org-home) is kept", async () => {
    let sentTeamHeader: string | null = "unset";
    const createSpy = vi.fn();
    server.use(
      getPostV1CreateNewApiKeyMockHandler((info) => {
        createSpy();
        sentTeamHeader = info.request.headers.get(TEAM_HEADER_NAME);
        return CREATED;
      }),
    );

    render(<CreateAPIKeyDialog open onOpenChange={() => {}} />);

    await userEvent.type(screen.getByLabelText("Name"), "CI key");
    fireEvent.click(screen.getAllByRole("checkbox")[0]);
    await userEvent.click(screen.getByRole("button", { name: "Create Key" }));

    await waitFor(() => expect(createSpy).toHaveBeenCalledTimes(1));
    expect(sentTeamHeader).toBeNull();
  });
});
