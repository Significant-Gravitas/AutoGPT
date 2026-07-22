import { getPostV2AddMarketplaceAgentMockHandler } from "@/app/api/__generated__/endpoints/library/library.msw";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  CreateSurface,
  getLastUsedTeam,
  setLastUsedTeam,
} from "@/components/contextual/TeamPicker/helpers";
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
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AddToLibraryButton } from "./AddToLibraryButton";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isLoggedIn: true, isUserLoading: false }),
}));

vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent: vi.fn() },
}));

const TEAM_A = {
  id: "team-a",
  name: "Growth",
  slug: "growth",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

const TEAM_B = {
  id: "team-b",
  name: "Design",
  slug: "design",
  isDefault: false,
  joinPolicy: "closed",
  orgId: "org-1",
};

const LIB_AGENT = {
  id: "lib-1",
  name: "Test Agent",
} as unknown as LibraryAgent;

function seedTeams(teams: (typeof TEAM_A)[]) {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID: null,
    orgs: [],
    teams,
    isLoaded: true,
  });
}

function renderButton() {
  return render(
    <AddToLibraryButton
      creatorSlug="creator"
      agentSlug="agent"
      agentName="Test Agent"
      agentGraphID="graph-1"
      isInLibrary={false}
    />,
  );
}

function captureAddHeader() {
  const state = { called: 0, teamHeader: "unset" as string | null };
  server.use(
    getPostV2AddMarketplaceAgentMockHandler((info) => {
      state.called += 1;
      state.teamHeader = info.request.headers.get(TEAM_HEADER_NAME);
      return LIB_AGENT;
    }),
  );
  return state;
}

beforeEach(() => {
  window.localStorage.clear();
  useOrgTeamStore.setState({
    activeOrgID: null,
    activeTeamID: null,
    orgs: [],
    teams: [],
    isLoaded: false,
  });
});

describe("AddToLibraryButton", () => {
  it("renders the plain Add button with no caret for solo users", () => {
    seedTeams([]);
    renderButton();

    const button = screen.getByRole("button", {
      name: "Add Test Agent to library",
    }) as HTMLButtonElement;
    expect(button.disabled).toBe(false);
    expect(
      screen.queryByRole("button", { name: /Choose where to add/i }),
    ).toBeNull();
  });

  it("disables the Add button until the org/team store has loaded", () => {
    // Default beforeEach leaves isLoaded=false: a team member must not be able
    // to click the solo control (which adds to org context) during the load.
    renderButton();

    const button = screen.getByRole("button", {
      name: "Add Test Agent to library",
    }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    expect(
      screen.queryByRole("button", { name: /Choose where to add/i }),
    ).toBeNull();
  });

  it("does not persist the target when the add request fails", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    let addCalls = 0;
    server.use(
      http.post("http://localhost:3000/api/proxy/api/library/agents", () => {
        addCalls += 1;
        return HttpResponse.json({ detail: "boom" }, { status: 500 });
      }),
    );

    renderButton();
    // No last-used yet, so the primary targets the Organization.
    await userEvent.click(
      screen.getByRole("button", { name: "Add Test Agent to Organization" }),
    );

    await waitFor(() => expect(addCalls).toBe(1));
    // A failed add must not update the remembered target.
    expect(getLastUsedTeam(CreateSurface.MarketplaceAdd)).toBeNull();
  });

  it("primary action adds to the last-used team via the X-Team-Id header", async () => {
    seedTeams([TEAM_A]);
    setLastUsedTeam(CreateSurface.MarketplaceAdd, TEAM_A.id);
    const add = captureAddHeader();

    renderButton();
    await userEvent.click(
      screen.getByRole("button", { name: "Add Test Agent to Growth" }),
    );

    await waitFor(() => expect(add.called).toBe(1));
    expect(add.teamHeader).toBe(TEAM_A.id);
  });

  it("menu selection adds with the chosen team and persists it as last-used", async () => {
    seedTeams([TEAM_A, TEAM_B]);
    const add = captureAddHeader();

    renderButton();
    // No last-used yet, so the primary targets the Organization.
    expect(
      screen.getByRole("button", { name: "Add Test Agent to Organization" }),
    ).toBeDefined();

    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Choose where to add Test Agent" }),
      { button: 0 },
    );
    fireEvent.click(
      await screen.findByRole("menuitem", { name: "Add to Design" }),
    );

    await waitFor(() => expect(add.called).toBe(1));
    expect(add.teamHeader).toBe(TEAM_B.id);
    expect(getLastUsedTeam(CreateSurface.MarketplaceAdd)).toBe(TEAM_B.id);
  });
});
