import {
  getGetV2ListUserOrganizationsMockHandler,
  getGetV2ListUserOrganizationsMockHandler401,
} from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { AccountMenuOrgList } from "@/components/layout/Navbar/components/AccountMenu/components/AccountMenuOrgList";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { Key, storage } from "@/services/storage/local-storage";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { useAuthMock } = vi.hoisted(() => ({
  useAuthMock: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: useAuthMock,
}));

import OrgTeamProvider from "../OrgTeamProvider";

const PERSONAL_ORG: OrgResponse = {
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  avatar_url: "https://example.com/jane.png",
  description: null,
  is_personal: true,
  member_count: 1,
  created_at: new Date("2026-01-05T00:00:00Z"),
};

const COMPANY_ORG: OrgResponse = {
  ...PERSONAL_ORG,
  id: "org-company",
  name: "Acme Inc",
  slug: "acme",
  avatar_url: "https://example.com/acme.png",
  is_personal: false,
  member_count: 12,
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

function serveOrgs(orgs: OrgResponse[]) {
  const requested = vi.fn();
  server.use(
    getGetV2ListUserOrganizationsMockHandler(() => {
      requested();
      return orgs;
    }),
  );
  return requested;
}

function renderProvider() {
  render(
    <OrgTeamProvider>
      <span>app content</span>
    </OrgTeamProvider>,
  );
}

async function waitForOrgs() {
  await waitFor(() => {
    expect(useOrgTeamStore.getState().isLoaded).toBe(true);
  });
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

afterEach(() => {
  resetClientRequestFailureLog();
  vi.clearAllMocks();
});

describe("OrgTeamProvider", () => {
  it("renders children and defaults the active org to the personal org on login", async () => {
    mockLoggedIn();
    const requested = serveOrgs([COMPANY_ORG, PERSONAL_ORG]);

    renderProvider();

    expect(screen.getByText("app content")).toBeDefined();
    await waitForOrgs();

    expect(requested).toHaveBeenCalledTimes(1);
    const state = useOrgTeamStore.getState();
    expect(state.orgs).toEqual([COMPANY_ORG, PERSONAL_ORG]);
    expect(state.activeOrgID).toBe(PERSONAL_ORG.id);
  });

  it("falls back to the first org when the user has no personal org", async () => {
    mockLoggedIn();
    serveOrgs([COMPANY_ORG]);

    renderProvider();

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBe(COMPANY_ORG.id);
    });
  });

  it("keeps a previously selected org instead of overriding with the default", async () => {
    window.localStorage.setItem("active-org-id", COMPANY_ORG.id);
    useOrgTeamStore.setState({ activeOrgID: COMPANY_ORG.id });
    mockLoggedIn();
    serveOrgs([COMPANY_ORG, PERSONAL_ORG]);

    renderProvider();

    await waitForOrgs();
    expect(useOrgTeamStore.getState().activeOrgID).toBe(COMPANY_ORG.id);
  });

  it("still marks the store loaded when the org fetch fails (UI must not hang)", async () => {
    mockLoggedIn();
    server.use(getGetV2ListUserOrganizationsMockHandler401());

    renderProvider();

    await waitForOrgs();
    expect(useOrgTeamStore.getState().orgs).toEqual([]);
    expect(useOrgTeamStore.getState().activeOrgID).toBeNull();
  });

  it("still marks the store loaded when the org fetch throws (network error)", async () => {
    mockLoggedIn();
    server.use(http.get("/api/proxy/api/orgs", () => HttpResponse.error()));

    renderProvider();

    await waitForOrgs();
    expect(useOrgTeamStore.getState().orgs).toEqual([]);
    expect(useOrgTeamStore.getState().activeOrgID).toBeNull();
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
    const requested = serveOrgs([]);

    renderProvider();

    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    expect(useOrgTeamStore.getState().isLoaded).toBe(true);
    expect(requested).not.toHaveBeenCalled();
  });

  it("clears org/team context on logout", async () => {
    useOrgTeamStore.setState({
      activeOrgID: PERSONAL_ORG.id,
      orgs: [PERSONAL_ORG],
      isLoaded: true,
    });
    mockLoggedOut();
    const requested = serveOrgs([]);

    renderProvider();

    await waitFor(() => {
      expect(useOrgTeamStore.getState().activeOrgID).toBeNull();
    });
    expect(useOrgTeamStore.getState().orgs).toEqual([]);
    expect(useOrgTeamStore.getState().isLoaded).toBe(false);
    expect(requested).not.toHaveBeenCalled();
  });
});

// The org list exactly as GET /api/orgs sends it, read by the real account
// menu: the personal org is not the user's first membership.
describe("the account menu once the org list loads", () => {
  it("starts in the personal org, marked Personal and showing its avatar", async () => {
    mockLoggedIn();
    serveOrgs([COMPANY_ORG, PERSONAL_ORG]);

    render(
      <OrgTeamProvider>
        <AccountMenuOrgList />
      </OrgTeamProvider>,
    );

    const personalRow = (await screen.findByText(PERSONAL_ORG.name)).closest(
      "button",
    );
    const companyRow = screen.getByText(COMPANY_ORG.name).closest("button");
    if (!personalRow || !companyRow) throw new Error("org rows not rendered");

    await waitForOrgs();
    expect(useOrgTeamStore.getState().activeOrgID).toBe(PERSONAL_ORG.id);
    expect(storage.get(Key.ACTIVE_ORG)).toBe(PERSONAL_ORG.id);

    expect(within(personalRow).getByText("Personal")).toBeDefined();
    expect(within(companyRow).queryByText("Personal")).toBeNull();

    await waitFor(() => {
      expect(personalRow.querySelector("img")?.getAttribute("src")).toBe(
        PERSONAL_ORG.avatar_url,
      );
    });
    expect(companyRow.querySelector("img")?.getAttribute("src")).toBe(
      COMPANY_ORG.avatar_url,
    );
  });
});
