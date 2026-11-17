import { customMutator } from "@/app/api/mutators/custom-mutator";
import type BackendAPI from "@/lib/autogpt-server-api/client";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import { ImpersonationState } from "@/lib/impersonation";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { InfiniteQueryObserver, QueryObserver } from "@tanstack/react-query";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { User } from "../../types";

// Only the server actions are faked. The store, the impersonation state, the
// request mutator and the browser query client all run for real, so these
// tests see the headers the page's own queries go out with.
const actions = vi.hoisted(() => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  refreshSession: vi.fn(),
  serverLogout: vi.fn(),
}));
vi.mock("../../actions", () => actions);

import { useAuthStore } from "../useAuthStore";

const admin = {
  id: "admin-1",
  email: "admin@example.com",
  role: "admin",
} as User;

type SentRequest = { url: string; actingAs: string | undefined };

let sent: SentRequest[] = [];

function makeRouter() {
  return {
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
  } as unknown as AppRouterInstance;
}

// Answers after a short delay, so the requests a page sends on load are still
// in flight when the signed-in user arrives. The body names whoever the
// request was made for.
async function fakeBackend(url: string, init?: RequestInit) {
  const headers = (init?.headers ?? {}) as Record<string, string>;
  const actingAs = headers[IMPERSONATION_HEADER_NAME];
  sent.push({ url, actingAs });
  await new Promise((resolve) => setTimeout(resolve, 20));
  return new Response(JSON.stringify({ owner: actingAs ?? "caller" }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

// The Library page's agent list and the orgs list: neither waits for sign-in.
function watchPageQueries() {
  const queryClient = getQueryClient();
  const library = new InfiniteQueryObserver(queryClient, {
    queryKey: ["/api/library/agents"],
    queryFn: () => customMutator("/api/library/agents", { method: "GET" }),
    initialPageParam: 1,
    getNextPageParam: () => undefined,
    refetchOnMount: false,
    staleTime: 5 * 60 * 1000,
  });
  const orgs = new QueryObserver(queryClient, {
    queryKey: ["/api/orgs"],
    queryFn: () => customMutator("/api/orgs", { method: "GET" }),
  });
  const stopLibrary = library.subscribe(() => {});
  const stopOrgs = orgs.subscribe(() => {});
  return {
    library,
    orgs,
    stop() {
      stopLibrary();
      stopOrgs();
    },
  };
}

function signIn(user: User) {
  actions.getCurrentUser.mockResolvedValue({ user });
  return useAuthStore.getState().initialize({
    api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
    router: makeRouter(),
    path: "/library",
  });
}

function sentTo(path: string) {
  return sent.filter((request) => request.url.endsWith(path));
}

beforeEach(() => {
  sent = [];
  localStorage.clear();
  sessionStorage.clear();
  ImpersonationState.clear();
  vi.stubGlobal("fetch", vi.fn(fakeBackend));
});

afterEach(() => {
  useAuthStore.getState().cleanup();
  useAuthStore.setState({
    user: null,
    hasLoadedUser: false,
    isUserLoading: true,
    isValidating: false,
    lastValidation: 0,
    routerRef: null,
    currentPath: "",
  });
  getQueryClient().clear();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("page queries started before the admin's session is known", () => {
  it("are sent again as the impersonated user once the admin is signed in", async () => {
    ImpersonationState.set("user-a", admin.id);
    const page = watchPageQueries();

    await signIn(admin);

    await vi.waitFor(() => {
      expect(page.library.getCurrentResult().data?.pages[0]).toMatchObject({
        data: { owner: "user-a" },
      });
      expect(page.orgs.getCurrentResult().data).toMatchObject({
        data: { owner: "user-a" },
      });
    });
    // The first requests left before the session was known, without the
    // target; the ones the page ends up showing carry it.
    expect(sentTo("/api/library/agents")[0].actingAs).toBeUndefined();
    expect(sentTo("/api/library/agents").at(-1)?.actingAs).toBe("user-a");
    expect(sentTo("/api/orgs").at(-1)?.actingAs).toBe("user-a");
    page.stop();
  });

  it("are not sent again when nobody is being impersonated", async () => {
    const page = watchPageQueries();

    await signIn(admin);

    await vi.waitFor(() => {
      expect(page.library.getCurrentResult().data?.pages[0]).toMatchObject({
        data: { owner: "caller" },
      });
    });
    expect(sentTo("/api/library/agents")).toHaveLength(1);
    expect(sentTo("/api/orgs")).toHaveLength(1);
    expect(sent.every((request) => request.actingAs === undefined)).toBe(true);
    page.stop();
  });
});
