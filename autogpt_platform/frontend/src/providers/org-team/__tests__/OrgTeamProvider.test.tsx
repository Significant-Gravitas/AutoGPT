import { useOrgTeamStore } from "@/services/org-team/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { useAuthMock } = vi.hoisted(() => ({
  useAuthMock: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: useAuthMock,
}));

import OrgTeamProvider from "../OrgTeamProvider";

// GET /api/orgs responds in snake_case — mocks must match the wire shape.
const PERSONAL_ORG_API = {
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  avatar_url: null,
  description: null,
  is_personal: true,
  member_count: 1,
  created_at: "2026-01-01T00:00:00Z",
};

const COMPANY_ORG_API = {
  id: "org-company",
  name: "Acme Inc",
  slug: "acme",
  avatar_url: "https://example.com/acme.png",
  description: null,
  is_personal: false,
  member_count: 12,
  created_at: "2026-01-01T00:00:00Z",
};

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
  avatarUrl: "https://example.com/acme.png",
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
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "true";
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
    delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it.each(["false", undefined])(
    "uses the server's own personal workspace and clears a saved shared scope when the flag is %s",
    async (flagValue) => {
      if (flagValue === undefined) {
        delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
      } else {
        process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = flagValue;
      }
      mockLoggedIn();
      useOrgTeamStore.setState({
        activeOrgID: COMPANY_ORG.id,
        activeTeamID: "team-company",
        orgs: [COMPANY_ORG, { ...PERSONAL_ORG, id: "somebody-elses-personal" }],
        isLoaded: true,
      });
      window.localStorage.setItem("unrelated-preference", "preserved");
      const resetQueries = vi.spyOn(getQueryClient(), "resetQueries");
      const fetchMock = vi.fn().mockImplementation((url: string) =>
        Promise.resolve({
          ok: true,
          json: async () => ({
            data: url.endsWith("/default")
              ? PERSONAL_ORG_API
              : [
                  {
                    id: "team-default",
                    name: "General",
                    slug: "general",
                    is_default: true,
                    join_policy: "OPEN",
                    org_id: PERSONAL_ORG.id,
                    is_member: true,
                  },
                ],
          }),
        }),
      );
      vi.stubGlobal("fetch", fetchMock);

      render(
        <OrgTeamProvider>
          <span>personal content</span>
        </OrgTeamProvider>,
      );
      expect(screen.queryByText("personal content")).toBeNull();
      await screen.findByText("personal content");

      expect(useOrgTeamStore.getState()).toMatchObject({
        activeOrgID: PERSONAL_ORG.id,
        activeTeamID: "team-default",
        orgs: [PERSONAL_ORG],
        isLoaded: true,
      });
      expect(
        fetchMock.mock.calls
          .map(([url]) => url)
          .filter((url) => url.startsWith("/api/proxy/api/orgs")),
      ).toEqual([
        "/api/proxy/api/orgs/default",
        `/api/proxy/api/orgs/${PERSONAL_ORG.id}/workspaces`,
      ]);
      expect(resetQueries).toHaveBeenCalled();
      expect(window.localStorage.getItem("unrelated-preference")).toBe(
        "preserved",
      );
      resetQueries.mockRestore();
    },
  );

  it("does not restore a cached shared workspace if the personal lookup fails", async () => {
    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "false";
    mockLoggedIn();
    useOrgTeamStore.setState({
      activeOrgID: COMPANY_ORG.id,
      orgs: [COMPANY_ORG],
      isLoaded: true,
    });
    mockOrgsResponse(null, false);

    render(
      <OrgTeamProvider>
        <span>resource actions</span>
      </OrgTeamProvider>,
    );
    await screen.findByText("We couldn't load your workspace.");

    expect(screen.queryByText("resource actions")).toBeNull();
    expect(useOrgTeamStore.getState().activeOrgID).toBeNull();
    expect(useOrgTeamStore.getState().orgs).toEqual([]);
  });

  it("discards in-flight shared-org results when the flag turns off", async () => {
    mockLoggedIn();
    useOrgTeamStore.setState({ activeOrgID: COMPANY_ORG.id });
    let finishOldRequest: ((value: unknown) => void) | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((url: string) => {
        if (url === "/api/proxy/api/orgs") {
          return new Promise((resolve) => {
            finishOldRequest = resolve;
          });
        }
        return Promise.resolve({
          ok: true,
          json: async () => ({
            data: url.endsWith("/default")
              ? PERSONAL_ORG_API
              : [
                  {
                    id: "team-default",
                    name: "General",
                    slug: "general",
                    is_default: true,
                    join_policy: "OPEN",
                    org_id: PERSONAL_ORG.id,
                    is_member: true,
                  },
                ],
          }),
        });
      }),
    );
    const { rerender } = render(
      <OrgTeamProvider>
        <span>resource actions</span>
      </OrgTeamProvider>,
    );

    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "false";
    rerender(
      <OrgTeamProvider>
        <span>resource actions</span>
      </OrgTeamProvider>,
    );
    await waitFor(() =>
      expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id),
    );
    finishOldRequest?.({
      ok: true,
      json: async () => ({ data: [COMPANY_ORG_API] }),
    });
    await screen.findByText("resource actions");

    expect(useOrgTeamStore.getState().orgs).toEqual([PERSONAL_ORG]);
    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
  });

  it("restores personal scope when a pending mutation selects a shared org after the flag turns off", async () => {
    mockLoggedIn();
    useOrgTeamStore.setState({ activeOrgID: COMPANY_ORG.id });
    const fetchMock = vi.fn().mockImplementation((url: string) =>
      Promise.resolve({
        ok: true,
        json: async () => ({
          data: url.endsWith("/default")
            ? PERSONAL_ORG_API
            : isWorkspacesUrl(url)
              ? [
                  {
                    id: "team-default",
                    name: "General",
                    slug: "general",
                    is_default: true,
                    join_policy: "OPEN",
                    org_id: PERSONAL_ORG.id,
                    is_member: true,
                  },
                ]
              : [COMPANY_ORG_API, PERSONAL_ORG_API],
        }),
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    let finishMutation!: () => void;
    const pendingMutation = new Promise<void>((resolve) => {
      finishMutation = resolve;
    }).then(() => useOrgTeamStore.getState().setActiveOrg(COMPANY_ORG.id));
    const { rerender } = render(
      <OrgTeamProvider>
        <span>resource actions</span>
      </OrgTeamProvider>,
    );

    process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS = "false";
    rerender(
      <OrgTeamProvider>
        <span>resource actions</span>
      </OrgTeamProvider>,
    );
    await screen.findByText("resource actions");
    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    fetchMock.mockClear();

    await act(async () => {
      finishMutation();
      await pendingMutation;
    });
    await screen.findByText("resource actions");
    expect(useOrgTeamStore.getState()).toMatchObject({
      activeOrgID: PERSONAL_ORG.id,
      activeTeamID: "team-default",
      isLoaded: true,
    });
    expect(fetchMock).not.toHaveBeenCalledWith(
      `/api/proxy/api/orgs/${COMPANY_ORG.id}/workspaces`,
      expect.anything(),
    );

    act(() => useOrgTeamStore.getState().setActiveTeam("stale-shared-team"));
    await screen.findByText("resource actions");
    expect(useOrgTeamStore.getState().activeTeamID).toBe("team-default");
    expect(window.localStorage.getItem("active-org-id")).toBe(PERSONAL_ORG.id);
    expect(window.localStorage.getItem("active-team-id")).toBe("team-default");
  });

  it("renders children, normalizes the snake_case response, and defaults the active org to the personal org on login", async () => {
    mockLoggedIn();
    const fetchMock = mockOrgsResponse([COMPANY_ORG_API, PERSONAL_ORG_API]);

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
                  is_member: true,
                  member_count: 3,
                  created_at: "2026-01-01T00:00:00Z",
                },
                {
                  id: "team-private",
                  name: "Secret",
                  slug: "secret",
                  description: null,
                  is_default: false,
                  join_policy: "PRIVATE",
                  org_id: PERSONAL_ORG.id,
                  is_member: false,
                  member_count: 2,
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
        headers: {
          "Content-Type": "application/json",
          "X-Org-Id": PERSONAL_ORG.id,
          "X-Team-Id": "",
        },
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
    mockOrgsResponse([COMPANY_ORG_API]);

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
    mockOrgsResponse([COMPANY_ORG_API, PERSONAL_ORG_API]);

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

  it("replaces a stored organization the user can no longer access", async () => {
    window.localStorage.setItem("active-org-id", "org-stale");
    useOrgTeamStore.setState({ activeOrgID: "org-stale" });
    mockLoggedIn();
    mockOrgsResponse([COMPANY_ORG_API, PERSONAL_ORG_API]);

    render(
      <OrgTeamProvider>
        <span>app content</span>
      </OrgTeamProvider>,
    );

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
      expect(useOrgTeamStore.getState().isLoaded).toBe(true);
    });
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
