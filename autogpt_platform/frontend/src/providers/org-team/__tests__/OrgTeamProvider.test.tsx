import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { useSupabaseMock } = vi.hoisted(() => ({
  useSupabaseMock: vi.fn(),
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: useSupabaseMock,
}));

import OrgTeamProvider from "../OrgTeamProvider";

const PERSONAL_ORG = {
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  avatarUrl: null,
  isPersonal: true,
  memberCount: 1,
};

const COMPANY_ORG = {
  id: "org-company",
  name: "Acme Inc",
  slug: "acme",
  avatarUrl: null,
  isPersonal: false,
  memberCount: 12,
};

function mockLoggedIn() {
  useSupabaseMock.mockReturnValue({
    isLoggedIn: true,
    user: { id: "user-1" },
  });
}

function mockLoggedOut() {
  useSupabaseMock.mockReturnValue({ isLoggedIn: false, user: null });
}

function mockOrgsResponse(orgs: unknown, ok = true) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    json: async () => ({ data: orgs }),
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("OrgTeamProvider", () => {
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

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("renders children and defaults the active org to the personal org on login", async () => {
    mockLoggedIn();
    const fetchMock = mockOrgsResponse([COMPANY_ORG, PERSONAL_ORG]);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    expect(screen.getByText("app content")).toBeDefined();
    await waitFor(() => {
      expect(useOrgTeamStore.getState().isLoaded).toBe(true);
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/proxy/api/orgs",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
    const state = useOrgTeamStore.getState();
    expect(state.orgs).toEqual([COMPANY_ORG, PERSONAL_ORG]);
    expect(state.activeOrgID).toBe(PERSONAL_ORG.id);
  });

  it("falls back to the first org when the user has no personal org", async () => {
    mockLoggedIn();
    mockOrgsResponse([COMPANY_ORG]);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBe(COMPANY_ORG.id);
    });
  });

  it("keeps a previously selected org instead of overriding with the default", async () => {
    window.localStorage.setItem("active-org-id", COMPANY_ORG.id);
    useOrgTeamStore.setState({ activeOrgID: COMPANY_ORG.id });
    mockLoggedIn();
    mockOrgsResponse([COMPANY_ORG, PERSONAL_ORG]);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().isLoaded).toBe(true);
    });
    expect(useOrgTeamStore.getState().activeOrgID).toBe(COMPANY_ORG.id);
  });

  it("still marks the store loaded when the org fetch fails (UI must not hang)", async () => {
    mockLoggedIn();
    mockOrgsResponse(null, false);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().isLoaded).toBe(true);
    });
    expect(useOrgTeamStore.getState().orgs).toEqual([]);
    expect(useOrgTeamStore.getState().activeOrgID).toBeNull();
  });

  it("still marks the store loaded when the org fetch throws (network error)", async () => {
    mockLoggedIn();
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().isLoaded).toBe(true);
    });
  });

  it("clears org/team context on logout", async () => {
    useOrgTeamStore.setState({
      activeOrgID: PERSONAL_ORG.id,
      orgs: [PERSONAL_ORG],
      isLoaded: true,
    });
    mockLoggedOut();
    const fetchMock = mockOrgsResponse([]);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBeNull();
    });
    expect(useOrgTeamStore.getState().orgs).toEqual([]);
    expect(useOrgTeamStore.getState().isLoaded).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
