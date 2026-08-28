import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
  getPostV2AddMarketplaceAgentMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2GetSpecificAgentResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  CreateSurface,
  getLastUsedTeam,
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
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AddToLibraryButton } from "./AddToLibraryButton";

const toastMocks = vi.hoisted(() => ({
  dismiss: vi.fn(),
  toast: vi.fn(),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({
    toast: toastMocks.toast.mockImplementation(() => ({
      dismiss: toastMocks.dismiss,
    })),
  }),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, isUserLoading: false }),
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
  graph_id: "graph-1",
  organization_id: "org-1",
  team_id: null,
} as unknown as LibraryAgent;

function seedTeams(
  teams: (typeof TEAM_A)[],
  activeTeamID: string | null = null,
) {
  useOrgTeamStore.setState({
    activeOrgID: "org-1",
    activeTeamID,
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
  toastMocks.toast.mockClear();
  toastMocks.dismiss.mockClear();
  window.localStorage.clear();
  useOrgTeamStore.setState({
    activeOrgID: null,
    activeTeamID: null,
    orgs: [],
    teams: [],
    isLoaded: false,
  });
  server.use(
    getGetV2ListLibraryAgentsMockHandler(
      getGetV2ListLibraryAgentsResponseMock({ agents: [] }),
    ),
    getGetV2GetSpecificAgentMockHandler(
      getGetV2GetSpecificAgentResponseMock({
        store_listing_version_id: "store-version-1",
      }),
    ),
  );
});

describe("AddToLibraryButton", () => {
  it("renders the plain Add button with no caret for solo users", async () => {
    seedTeams([]);
    renderButton();

    const button = screen.getByRole("button", {
      name: "Add Test Agent to library",
    }) as HTMLButtonElement;
    await waitFor(() => expect(button.disabled).toBe(false));
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
    const addButton = screen.getByRole("button", {
      name: "Add Test Agent to Organization",
    });
    await waitFor(() =>
      expect((addButton as HTMLButtonElement).disabled).toBe(false),
    );
    await userEvent.click(addButton);

    await waitFor(() => expect(addCalls).toBe(1));
    // A failed add must not update the remembered target.
    expect(getLastUsedTeam("org-1", CreateSurface.MarketplaceAdd)).toBeNull();
  });

  it("primary action adds to the last-used team via the X-Team-Id header", async () => {
    seedTeams([TEAM_A]);
    setLastUsedTeam("org-1", CreateSurface.MarketplaceAdd, TEAM_A.id);
    const add = captureAddHeader();

    renderButton();
    const addButton = screen.getByRole("button", {
      name: "Add Test Agent to Growth",
    });
    await waitFor(() =>
      expect((addButton as HTMLButtonElement).disabled).toBe(false),
    );
    await userEvent.click(addButton);

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
    expect(getLastUsedTeam("org-1", CreateSurface.MarketplaceAdd)).toBe(
      TEAM_B.id,
    );
  });

  it("offers only destinations that do not already contain the graph", async () => {
    seedTeams([TEAM_A, TEAM_B], TEAM_A.id);
    server.use(
      getGetV2ListLibraryAgentsMockHandler(
        getGetV2ListLibraryAgentsResponseMock({
          agents: [
            { ...LIB_AGENT, id: "lib-home", team_id: null },
            { ...LIB_AGENT, id: "lib-a", team_id: TEAM_A.id },
          ],
        }),
      ),
    );

    renderButton();

    expect(
      await screen.findByRole("button", { name: "Add Test Agent to Design" }),
    ).toHaveProperty("disabled", false);
    fireEvent.pointerDown(
      screen.getByRole("button", { name: "Choose where to add Test Agent" }),
      { button: 0 },
    );
    expect(
      screen.queryByRole("menuitem", { name: "Add to Organization" }),
    ).toBeNull();
    expect(
      screen.queryByRole("menuitem", { name: "Add to Growth" }),
    ).toBeNull();
  });

  it("undoes an install with the returned destination scope after nav changes", async () => {
    seedTeams([TEAM_A, TEAM_B], TEAM_A.id);
    setLastUsedTeam("org-1", CreateSurface.MarketplaceAdd, TEAM_B.id);
    const installedInB = {
      ...LIB_AGENT,
      id: "lib-b",
      team_id: TEAM_B.id,
    } as LibraryAgent;
    let deleteHeaders: {
      organization: string | null;
      team: string | null;
    } | null = null;
    server.use(
      getPostV2AddMarketplaceAgentMockHandler(installedInB),
      http.delete(
        "http://localhost:3000/api/proxy/api/library/agents/lib-b",
        ({ request }) => {
          deleteHeaders = {
            organization: request.headers.get(ORG_HEADER_NAME),
            team: request.headers.get(TEAM_HEADER_NAME),
          };
          return new HttpResponse(null, { status: 204 });
        },
      ),
    );

    renderButton();
    const addButton = await screen.findByRole("button", {
      name: "Add Test Agent to Design",
    });
    await waitFor(() =>
      expect((addButton as HTMLButtonElement).disabled).toBe(false),
    );
    await userEvent.click(addButton);
    await waitFor(() => expect(toastMocks.toast).toHaveBeenCalled());
    const addedToast = toastMocks.toast.mock.calls.find(
      ([options]) =>
        options.title === "Agent Test Agent added to your library.",
    )?.[0];
    expect(addedToast).toBeDefined();
    render(addedToast!.description);
    useOrgTeamStore.setState({ activeTeamID: null });
    await userEvent.click(await screen.findByRole("button", { name: "Undo" }));

    await waitFor(() =>
      expect(deleteHeaders).toEqual({
        organization: "org-1",
        team: TEAM_B.id,
      }),
    );
  });
});
