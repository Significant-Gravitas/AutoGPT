import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { useAuthMock } = vi.hoisted(() => ({
  useAuthMock: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: useAuthMock,
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
  useAuthMock.mockReturnValue({
    isLoggedIn: true,
    user: { id: "user-1" },
    isUserLoading: false,
  });
}

function mockLoggedOut() {
  useAuthMock.mockReturnValue({
    isLoggedIn: false,
    user: null,
    isUserLoading: false,
  });
}

function mockSessionHydrating() {
  useAuthMock.mockReturnValue({
    isLoggedIn: false,
    user: null,
    isUserLoading: true,
  });
}

function isWorkspacesUrl(url: unknown) {
  return typeof url === "string" && url.includes("/workspaces");
}

// The provider fetches orgs first, then the active org's teams. Route by
// URL so a single stub serves both without teams polluting the org list.
function mockOrgsResponse(orgs: unknown, ok = true) {
  const fetchMock = vi.fn().mockImplementation((url: unknown) =>
    Promise.resolve({
      ok,
      json: async () => ({ data: isWorkspacesUrl(url) ? [] : orgs }),
    }),
  );
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

  it("fetches the active org's teams and maps them into the store (camelCase, no active team)", async () => {
    mockLoggedIn();
    const fetchMock = vi.fn().mockImplementation((url: unknown) =>
      Promise.resolve({
        ok: true,
        json: async () => ({
          data: isWorkspacesUrl(url)
            ? [
                {
                  id: "team-default",
                  name: "General",
                  slug: "general",
                  description: null,
                  is_default: true,
                  join_policy: "OPEN",
                  org_id: PERSONAL_ORG.id,
                  member_count: 3,
                  created_at: "2026-01-01T00:00:00Z",
                },
              ]
            : [PERSONAL_ORG],
        }),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().teams).toHaveLength(1);
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/proxy/api/orgs/${PERSONAL_ORG.id}/workspaces`,
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      }),
    );
    const [team] = useOrgTeamStore.getState().teams;
    expect(team).toEqual({
      id: "team-default",
      name: "General",
      slug: "general",
      isDefault: true,
      joinPolicy: "OPEN",
      orgId: PERSONAL_ORG.id,
    });
    // Teams are badges/filters now — the provider never auto-selects one.
    expect(useOrgTeamStore.getState().activeTeamID).toBeNull();
  });

  it("leaves teams empty when the teams fetch fails or errors", async () => {
    mockLoggedIn();
    const fetchMock = vi.fn().mockImplementation((url: unknown) => {
      if (isWorkspacesUrl(url)) {
        return Promise.reject(new Error("offline"));
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ data: [PERSONAL_ORG] }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    });
    expect(useOrgTeamStore.getState().teams).toEqual([]);
  });

  it("leaves teams empty when the teams endpoint responds not-ok", async () => {
    mockLoggedIn();
    const fetchMock = vi.fn().mockImplementation((url: unknown) =>
      Promise.resolve({
        ok: !isWorkspacesUrl(url),
        json: async () => ({ data: [PERSONAL_ORG] }),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    });
    expect(useOrgTeamStore.getState().teams).toEqual([]);
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

  it("keeps the stored org while the session is still hydrating", async () => {
    // Regression: isLoggedIn is transiently false during session
    // hydration. Clearing context then flips activeOrgID to null,
    // which wiped the query cache mid-flight and stranded every
    // in-flight page query in a forever-pending state (e2e: api-keys
    // list spinner never resolved).
    window.localStorage.setItem("active-org-id", PERSONAL_ORG.id);
    useOrgTeamStore.setState({
      activeOrgID: PERSONAL_ORG.id,
      orgs: [PERSONAL_ORG],
      isLoaded: true,
    });
    mockSessionHydrating();
    const fetchMock = mockOrgsResponse([]);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    expect(useOrgTeamStore.getState().isLoaded).toBe(true);
    expect(fetchMock).not.toHaveBeenCalled();
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
