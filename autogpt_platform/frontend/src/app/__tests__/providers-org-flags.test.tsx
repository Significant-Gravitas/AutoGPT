import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useOrgTeamStore } from "@/services/org-team/store";

const { passthrough, user, launchDarklyFlags } = vi.hoisted(() => ({
  passthrough: ({ children }: { children: React.ReactNode }) => children,
  user: { id: "flagged-user" },
  launchDarklyFlags: { value: {} as Record<string, boolean> },
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user, isLoggedIn: true, isUserLoading: false }),
}));
vi.mock("@/lib/autogpt-server-api/context", () => ({
  BackendAPIProvider: passthrough,
}));
vi.mock("@/providers/agent-credentials/credentials-provider", () => ({
  default: passthrough,
}));
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: passthrough,
}));
vi.mock("@/providers/posthog/posthog-provider", () => ({
  PostHogProvider: passthrough,
  PostHogPageViewTracker: () => null,
  PostHogUserTracker: () => null,
}));
vi.mock("@/components/monitor/SentryUserTracker", () => ({
  SentryUserTracker: () => null,
}));
vi.mock("@/services/analytics/AdsConversionTracker", () => ({
  AdsConversionTracker: () => null,
}));
vi.mock("next-themes", () => ({ ThemeProvider: passthrough }));
vi.mock("nuqs/adapters/next/app", () => ({ NuqsAdapter: passthrough }));
vi.mock("@/services/environment", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      areFeatureFlagsEnabled: () => true,
      getLaunchDarklyClientId: () => "test-client",
    },
  };
});
vi.mock("launchdarkly-react-client-sdk", async () => {
  const { createContext, useContext } = await import("react");
  const flags = createContext<Record<string, boolean>>({});
  return {
    LDProvider: ({ children }: { children: React.ReactNode }) => (
      <flags.Provider value={launchDarklyFlags.value}>
        {children}
      </flags.Provider>
    ),
    useFlags: () => useContext(flags),
  };
});

import { Providers } from "../providers";

beforeEach(() => {
  delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
  launchDarklyFlags.value = { SHOW_ORG_SETTINGS: true };
  useOrgTeamStore.getState().clearContext();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  useOrgTeamStore.getState().clearContext();
});

it.each(["enabled", "disabled", "timeout"])(
  "preserves saved shared scope while LaunchDarkly is pending, then resolves %s",
  async (resolution) => {
    launchDarklyFlags.value = {};
    if (resolution === "timeout") vi.useFakeTimers();
    useOrgTeamStore.getState().setActiveOrg("shared-org");
    useOrgTeamStore.getState().setActiveTeam("shared-team");
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      const personal = url.includes("personal-org") || url.endsWith("/default");
      const org = {
        id: personal ? "personal-org" : "shared-org",
        name: personal ? "Personal" : "Shared",
        slug: personal ? "personal" : "shared",
        is_personal: personal,
        avatar_url: null,
        member_count: personal ? 1 : 2,
      };
      return Promise.resolve({
        ok: true,
        json: async () => ({
          data: url.endsWith("/workspaces")
            ? [
                {
                  id: personal ? "default-team" : "shared-team",
                  name: personal ? "General" : "Shared team",
                  slug: "team",
                  is_default: personal,
                  join_policy: "OPEN",
                  org_id: org.id,
                  is_member: true,
                },
              ]
            : url.endsWith("/default")
              ? org
              : [org],
        }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = render(
      <Providers>
        <span>resource actions</span>
      </Providers>,
    );

    expect(screen.queryByText("resource actions")).toBeNull();
    expect(screen.getByRole("status").textContent).toContain(
      "Loading your workspace",
    );
    expect(fetchMock).not.toHaveBeenCalled();
    expect(useOrgTeamStore.getState()).toMatchObject({
      activeOrgID: "shared-org",
      activeTeamID: "shared-team",
    });
    expect(window.localStorage.getItem("active-org-id")).toBe("shared-org");
    expect(window.localStorage.getItem("active-team-id")).toBe("shared-team");

    if (resolution === "timeout") {
      await act(() => vi.advanceTimersByTimeAsync(5000));
      vi.useRealTimers();
    } else {
      launchDarklyFlags.value = { SHOW_ORG_SETTINGS: resolution === "enabled" };
      rerender(
        <Providers>
          <span>resource actions</span>
        </Providers>,
      );
    }
    await waitFor(() => expect(useOrgTeamStore.getState().isLoaded).toBe(true));
    await screen.findByText("resource actions");

    const enabled = resolution === "enabled";
    expect(useOrgTeamStore.getState()).toMatchObject({
      activeOrgID: enabled ? "shared-org" : "personal-org",
      activeTeamID: enabled ? "shared-team" : "default-team",
    });
    expect(fetchMock.mock.calls.map(([url]) => url)).toContain(
      enabled ? "/api/proxy/api/orgs" : "/api/proxy/api/orgs/default",
    );
    expect(fetchMock.mock.calls.map(([url]) => url)).not.toContain(
      enabled ? "/api/proxy/api/orgs/default" : "/api/proxy/api/orgs",
    );
  },
);

it("lets organization initialization observe the enabled LaunchDarkly key in the real provider tree", async () => {
  delete process.env.NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS;
  useOrgTeamStore.setState({
    activeOrgID: "shared-org",
    orgs: [],
    teams: [],
    isLoaded: false,
  });
  const fetchMock = vi.fn().mockImplementation((url: string) =>
    Promise.resolve({
      ok: true,
      json: async () => ({
        data: url.endsWith("/workspaces")
          ? []
          : [
              {
                id: "shared-org",
                name: "Shared",
                slug: "shared",
                is_personal: false,
                avatar_url: null,
                member_count: 2,
              },
            ],
      }),
    }),
  );
  vi.stubGlobal("fetch", fetchMock);

  render(
    <Providers>
      <span>app content</span>
    </Providers>,
  );
  await waitFor(() => expect(useOrgTeamStore.getState().isLoaded).toBe(true));

  expect(useOrgTeamStore.getState().activeOrgID).toBe("shared-org");
  expect(fetchMock.mock.calls.map(([url]) => url)).toContain(
    "/api/proxy/api/orgs",
  );
  expect(fetchMock.mock.calls.map(([url]) => url)).not.toContain(
    "/api/proxy/api/orgs/default",
  );
});
