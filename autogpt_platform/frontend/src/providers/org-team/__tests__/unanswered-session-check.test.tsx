import { getGetV2ListUserOrganizationsMockHandler } from "@/app/api/__generated__/endpoints/orgs/orgs.msw";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { SESSION_CHECK_DEADLINE_MS } from "@/lib/auth/hooks/session-check-deadline";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { Key, storage } from "@/services/storage/local-storage";
import { act, cleanup, render } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import OrgTeamProvider from "../OrgTeamProvider";

// Only the server actions are faked: the auth store, its session checks and
// their deadlines, and the provider run as shipped.
const actions = vi.hoisted(() => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  refreshSession: vi.fn(),
  serverLogout: vi.fn(),
}));
vi.mock("@/lib/auth/actions", () => actions);

const member: User = {
  id: "user-1",
  email: "member@example.com",
  role: "authenticated",
  user_metadata: {},
};

const PERSONAL_ORG: OrgResponse = {
  id: "org-personal",
  name: "Jane's org",
  slug: "jane",
  avatar_url: null,
  description: null,
  is_personal: true,
  member_count: 1,
  created_at: new Date("2026-01-05T00:00:00Z"),
};
const TEAM_ORG: OrgResponse = {
  ...PERSONAL_ORG,
  id: "org-acme",
  name: "Acme",
  slug: "acme",
  is_personal: false,
  member_count: 12,
};
const TEAM_ID = "team-design";

const signedIn = { user: member, isValid: true };
const signedOut = { user: null, isValid: false };
const networkError = new TypeError("Failed to fetch");
const never = () => new Promise<never>(() => {});

function AnotherPage() {
  useAuth();
  return null;
}

function renderApp() {
  function app(page: ReactNode) {
    return (
      <BackendAPIProvider>
        <OrgTeamProvider>{page}</OrgTeamProvider>
      </BackendAPIProvider>
    );
  }
  const { rerender } = render(app(null));
  // A page that mounts useAuth starts the store up again while nobody is
  // signed in, as a navigation does.
  return { openAnotherPage: () => rerender(app(<AnotherPage />)) };
}

// Runs a step, then the promise chains it starts, inside act: React renders,
// and the provider's effects run on, everything the store published meanwhile.
async function run<T>(step: () => T): Promise<T> {
  let result!: T;
  await act(async () => {
    result = step();
    await vi.advanceTimersByTimeAsync(0);
  });
  return result;
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

function holdNextAnswer(action: ReturnType<typeof vi.fn>) {
  let answer: (value: unknown) => void = () => {};
  action.mockReturnValueOnce(
    new Promise((resolve) => {
      answer = resolve;
    }),
  );
  return (value: unknown) => answer(value);
}

async function waitForOrgs() {
  const loaded = () => expect(useOrgTeamStore.getState().isLoaded).toBe(true);
  await act(() => vi.waitFor(loaded, { timeout: 5_000 }));
}

function expectTeamContext() {
  expect(useOrgTeamStore.getState()).toMatchObject({
    activeOrgID: TEAM_ORG.id,
    activeTeamID: TEAM_ID,
  });
  expect(storage.get(Key.ACTIVE_ORG)).toBe(TEAM_ORG.id);
  expect(storage.get(Key.ACTIVE_TEAM)).toBe(TEAM_ID);
}

function expectNoContext() {
  expect(useOrgTeamStore.getState()).toMatchObject({
    activeOrgID: null,
    activeTeamID: null,
  });
  expect(storage.get(Key.ACTIVE_ORG)).toBeNull();
  expect(storage.get(Key.ACTIVE_TEAM)).toBeNull();
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  vi.spyOn(console, "warn").mockImplementation(() => undefined);
  Object.values(actions).forEach((action) => action.mockReset());
  localStorage.clear();
  // A member working in a team org's team, the context a tab comes back to.
  storage.set(Key.ACTIVE_ORG, TEAM_ORG.id);
  storage.set(Key.ACTIVE_TEAM, TEAM_ID);
  useOrgTeamStore.setState({
    activeOrgID: TEAM_ORG.id,
    activeTeamID: TEAM_ID,
    orgs: [],
    teams: [],
    isLoaded: false,
  });
  server.use(
    getGetV2ListUserOrganizationsMockHandler([PERSONAL_ORG, TEAM_ORG]),
  );
});

afterEach(() => {
  cleanup();
  useAuthStore.getState().cleanup();
  useAuthStore.setState({
    user: null,
    hasLoadedUser: false,
    isUserLoading: true,
    isValidating: false,
    lastValidation: 0,
    routerRef: null,
    apiRef: null,
    currentPath: "",
    // A test that fails mid-start-up leaves that start-up waiting for good.
    // A new epoch and no pending start-up let the next test begin afresh.
    initializationPromise: null,
    sessionEpoch: useAuthStore.getState().sessionEpoch + 1,
  });
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("the active org and team at start-up (#17104)", () => {
  it("survive a lookup that can't reach the server, once the fallback finds the user", async () => {
    actions.getCurrentUser.mockRejectedValue(networkError);
    const answerFallback = holdNextAnswer(actions.validateSession);

    await run(renderApp);

    // The fallback is out; until it answers, nobody is known to be signed out.
    expect(actions.validateSession).toHaveBeenCalledTimes(1);
    expectTeamContext();
    expect(useAuthStore.getState().isUserLoading).toBe(true);

    await run(() => answerFallback(signedIn));
    await waitForOrgs();

    expect(useAuthStore.getState().user).toEqual(member);
    expectTeamContext();
  });

  it("survive a lookup that misses its deadline, once the fallback finds the user", async () => {
    actions.getCurrentUser.mockImplementation(never);
    const answerFallback = holdNextAnswer(actions.validateSession);

    await run(renderApp);
    await advance(SESSION_CHECK_DEADLINE_MS);

    expect(actions.validateSession).toHaveBeenCalledTimes(1);
    expectTeamContext();
    expect(useAuthStore.getState().isUserLoading).toBe(true);

    await run(() => answerFallback(signedIn));
    await waitForOrgs();

    expect(useAuthStore.getState().user).toEqual(member);
    expectTeamContext();
  });

  it("survive a start-up that nothing answers, until a retry finds the user", async () => {
    actions.getCurrentUser.mockImplementation(never);
    actions.validateSession
      .mockImplementationOnce(never)
      .mockResolvedValue(signedIn);

    await run(renderApp);
    await advance(SESSION_CHECK_DEADLINE_MS);
    expectTeamContext();
    expect(useAuthStore.getState().isUserLoading).toBe(true);
    await advance(SESSION_CHECK_DEADLINE_MS);

    // Start-up is over as "couldn't reach the backend": nobody signed in and
    // a backoff retry waiting, with the org and team kept.
    expect(useAuthStore.getState()).toMatchObject({
      user: null,
      isUserLoading: false,
    });
    expect(useAuthStore.getState().retryTimer).not.toBeNull();
    expectTeamContext();

    await advance(1_000);
    await waitForOrgs();

    expect(actions.validateSession).toHaveBeenCalledTimes(2);
    expect(useAuthStore.getState().user).toEqual(member);
    expectTeamContext();
  });

  it("go once a retry after such a start-up finds the session ended", async () => {
    actions.getCurrentUser.mockRejectedValue(networkError);
    actions.validateSession
      .mockRejectedValueOnce(networkError)
      .mockResolvedValue(signedOut);

    await run(renderApp);
    expectTeamContext();
    await advance(1_000);

    expect(actions.validateSession).toHaveBeenCalledTimes(2);
    expectNoContext();
  });

  it("go when a later start-up's fallback finds no session", async () => {
    actions.getCurrentUser.mockRejectedValue(networkError);
    actions.validateSession
      .mockRejectedValueOnce(networkError)
      .mockResolvedValue(signedOut);

    const app = await run(renderApp);
    expectTeamContext();
    await run(app.openAnotherPage);

    expect(actions.getCurrentUser).toHaveBeenCalledTimes(2);
    expectNoContext();
  });

  it("still go as soon as the lookup answers that nobody is signed in", async () => {
    actions.getCurrentUser.mockResolvedValue({ user: null });
    const answerFallback = holdNextAnswer(actions.validateSession);

    await run(renderApp);

    // An answer: the tab reads as signed out without waiting for the fallback.
    expect(useAuthStore.getState()).toMatchObject({
      user: null,
      isUserLoading: false,
    });
    expectNoContext();

    await run(() => answerFallback(signedOut));
    expectNoContext();
  });

  it("still go when the lookup can't reach the server and the fallback finds no session", async () => {
    actions.getCurrentUser.mockRejectedValue(networkError);
    actions.validateSession.mockResolvedValue(signedOut);

    await run(renderApp);

    expect(useAuthStore.getState()).toMatchObject({
      user: null,
      isUserLoading: false,
    });
    expectNoContext();
  });
});
