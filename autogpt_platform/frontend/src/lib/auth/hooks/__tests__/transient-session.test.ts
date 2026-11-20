import type BackendAPI from "@/lib/autogpt-server-api/client";
import { getSystemHeaders, ImpersonationState } from "@/lib/impersonation";
import { SessionKey } from "@/services/storage/session-storage";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { User } from "../../types";

// Only the server actions are faked; the store, auth helpers, impersonation
// state and header builder run as shipped.
const actions = vi.hoisted(() => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  refreshSession: vi.fn(),
  serverLogout: vi.fn(),
}));
vi.mock("../../actions", () => actions);
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({ clear: vi.fn() }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity: vi.fn(),
}));
const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast }));
const sentry = vi.hoisted(() => ({
  addBreadcrumb: vi.fn(),
  captureMessage: vi.fn(),
}));
vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  ...sentry,
}));

import {
  createRouterQueue,
  disposeRouterQueue,
} from "@/tests/integrations/next-router-queue";
import { SESSION_CHECK_DEADLINE_MS } from "../session-check-deadline";
import { useAuthStore } from "../useAuthStore";

const ACT_AS = "X-Act-As-User-Id";
const admin = {
  id: "admin-1",
  email: "admin@example.com",
  role: "admin",
} as User;

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

function validate(router = makeRouter()) {
  return useAuthStore
    .getState()
    .validateSession({ router, path: "/library", force: true });
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  ImpersonationState.clear();
  ImpersonationState.set("user-a", admin.id);
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
    apiRef: null,
    currentPath: "",
  });
  ImpersonationState.clear();
  vi.clearAllMocks();
});

describe("a session check that can't get an answer changes nothing", () => {
  it("keeps the admin and their target when the check never reaches the server", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    actions.validateSession.mockRejectedValue(new TypeError("Load failed"));
    const router = makeRouter();

    await validate(router);

    expect(useAuthStore.getState().user).toBe(admin);
    expect(router.push).not.toHaveBeenCalled();
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });

  it("keeps them when the auth backend reports itself unavailable", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    actions.validateSession.mockResolvedValue({
      user: null,
      isValid: false,
      isUnavailable: true,
    });

    await validate();

    expect(useAuthStore.getState().user).toBe(admin);
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });

  it("keeps the target when a tab starts up while the backend is unavailable", async () => {
    actions.getCurrentUser.mockResolvedValue({ user: null });
    actions.validateSession.mockResolvedValue({
      user: null,
      isValid: false,
      isUnavailable: true,
    });

    await useAuthStore.getState().initialize({
      api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
      router: makeRouter(),
      path: "/library",
    });

    // Kept for when the admin is known again; nothing is sent meanwhile,
    // because nobody is known to be signed in.
    expect(ImpersonationState.read()).toEqual({
      target: "user-a",
      owner: admin.id,
    });
    expect(getSystemHeaders()).toEqual({});
  });

  it("keeps the admin signed in when a refresh can't reach the backend", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    actions.refreshSession.mockResolvedValue({
      user: null,
      error: "connection reset",
      isUnavailable: true,
    });

    await useAuthStore.getState().refreshSession();

    expect(useAuthStore.getState().user).toBe(admin);
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });
});

describe("a deploy skew the reload didn't fix", () => {
  const reload = vi.fn();
  const nativeReload = Object.getOwnPropertyDescriptor(
    window.location,
    "reload",
  );

  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(window.location, "reload", {
      configurable: true,
      value: reload,
    });
  });

  afterEach(() => {
    if (nativeReload) {
      Object.defineProperty(window.location, "reload", nativeReload);
    }
    vi.useRealTimers();
  });

  it("asks for a refresh rather than reloading again when the retries outlast the cooldown", async () => {
    // This page is the one the skew reload loaded a moment ago.
    sessionStorage.setItem(
      SessionKey.DEPLOYMENT_RELOAD_AT,
      String(Date.now() - 5_000),
    );
    // Retries run against the page the store last saw, as in the app.
    useAuthStore.setState({
      user: admin,
      hasLoadedUser: true,
      routerRef: makeRouter(),
      currentPath: "/library",
    });
    actions.validateSession.mockRejectedValue(
      Object.assign(new Error("Server Action was not found on the server."), {
        name: "UnrecognizedActionError",
      }),
    );

    await validate();
    // Every retry of the backoff (1 + 2 + 4 + 8 + 16 + 30 s) meets the skew.
    await vi.advanceTimersByTimeAsync(120_000);

    expect(actions.validateSession).toHaveBeenCalledTimes(7);
    expect(reload).not.toHaveBeenCalled();
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "A new version of AutoGPT is available",
      }),
    );
    expect(useAuthStore.getState().user).toBe(admin);
  });
});

describe("a session check that never gets an answer (#16988)", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  afterEach(() => {
    disposeRouterQueue();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("lets the next refocus check the session once the deadline passes", async () => {
    // Signed in, with the session listeners running as in the app.
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    await useAuthStore.getState().initialize({
      api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
      router: makeRouter(),
      path: "/library",
    });
    // The first check's server action was dropped: it never answers.
    actions.validateSession.mockReturnValueOnce(new Promise(() => {}));
    actions.validateSession.mockResolvedValue({ user: admin, isValid: true });

    void validate();
    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS);
    // Unavailable, so a backoff retry now waits its first second.
    expect(useAuthStore.getState().retryTimer).not.toBeNull();
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.advanceTimersByTimeAsync(0);

    // The refocus checked at once, in place of that retry, which never fires.
    expect(actions.validateSession).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(5_000);
    expect(actions.validateSession).toHaveBeenCalledTimes(2);
    expect(useAuthStore.getState().user).toBe(admin);
  });

  it("ignores an answer that arrives after the deadline, and the next check finds the session ended", async () => {
    const router = makeRouter();
    // Retries run against the page the store last saw, as in the app.
    useAuthStore.setState({
      user: admin,
      hasLoadedUser: true,
      routerRef: router,
      currentPath: "/library",
    });
    const signedOut = {
      user: null,
      isValid: false,
      redirectPath: "/login?next=%2Flibrary",
    };
    let answerLate: (value: unknown) => void = () => {};
    actions.validateSession.mockReturnValueOnce(
      new Promise((resolve) => {
        answerLate = resolve;
      }),
    );
    actions.validateSession.mockResolvedValue(signedOut);

    void validate(router);
    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS);
    // The session ended meanwhile, and the check's answer comes too late.
    answerLate(signedOut);
    await vi.advanceTimersByTimeAsync(0);

    expect(useAuthStore.getState().user).toBe(admin);
    expect(router.push).not.toHaveBeenCalled();

    // The backoff retry a second later is the next check, and it signs out.
    await vi.advanceTimersByTimeAsync(1_000);
    expect(actions.validateSession).toHaveBeenCalledTimes(2);
    expect(useAuthStore.getState().user).toBeNull();
    expect(router.push).toHaveBeenCalledWith(signedOut.redirectPath);
  });

  it("acts on an answer that arrives just before the deadline", async () => {
    const router = makeRouter();
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    const signedOut = {
      user: null,
      isValid: false,
      redirectPath: "/login?next=%2Flibrary",
    };
    let answer: (value: unknown) => void = () => {};
    actions.validateSession.mockReturnValueOnce(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );

    void validate(router);
    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS - 1);
    answer(signedOut);
    await vi.advanceTimersByTimeAsync(0);

    expect(useAuthStore.getState().user).toBeNull();
    expect(router.push).toHaveBeenCalledWith(signedOut.redirectPath);
  });

  it("finishes start-up when its first lookup never answers", async () => {
    actions.getCurrentUser.mockReturnValueOnce(new Promise(() => {}));
    actions.validateSession.mockResolvedValue({ user: admin, isValid: true });

    void useAuthStore.getState().initialize({
      api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
      router: makeRouter(),
      path: "/library",
    });
    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS);

    // Start-up gave up on the lookup, asked the backend instead and let go
    // of the check, so the next refocus checks again.
    expect(useAuthStore.getState()).toMatchObject({
      user: admin,
      isValidating: false,
    });
    document.dispatchEvent(new Event("visibilitychange"));
    await vi.advanceTimersByTimeAsync(0);
    expect(actions.validateSession).toHaveBeenCalledTimes(2);
  });

  it("lets the next check go out when a refresh never answers", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    actions.refreshSession.mockReturnValueOnce(new Promise(() => {}));
    actions.validateSession.mockResolvedValue({ user: admin, isValid: true });
    let refreshed: unknown;
    void useAuthStore
      .getState()
      .refreshSession()
      .then((result) => {
        refreshed = result;
      });

    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS);

    // It counts as unavailable: the admin stays signed in, the check is let go.
    expect(refreshed).toMatchObject({ isUnavailable: true });
    expect(useAuthStore.getState()).toMatchObject({
      user: admin,
      isValidating: false,
    });
    await validate();
    expect(actions.validateSession).toHaveBeenCalledTimes(1);
  });

  // What the deadline can't do: Next runs one server action at a time (#17028).
  it("still queues the next check behind one the server took but never answered", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    const queue = createRouterQueue();
    const reached: string[] = [];
    actions.validateSession
      .mockImplementationOnce(() =>
        queue.callServer(() => {
          reached.push("first");
          return new Promise(() => {});
        }),
      )
      .mockImplementationOnce(() =>
        queue.callServer(async () => {
          reached.push("second");
          return { user: admin, isValid: true };
        }),
      );

    void validate();
    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS);
    // The store sends its next check, but Next holds it back, past its own
    // deadline too.
    const second = validate();
    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS);
    await second;
    expect(actions.validateSession).toHaveBeenCalledTimes(2);
    expect(reached).toEqual(["first"]);

    // A navigation discards the unanswered action, and the queue moves on.
    queue.navigate().arrive();
    await vi.waitFor(() => expect(reached).toEqual(["first", "second"]));
    expect(useAuthStore.getState().user).toBe(admin);
  });

  it("runs its deadline only while a check waits for its answer", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    let answer: (value: unknown) => void = () => {};
    actions.validateSession.mockReturnValueOnce(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    const baseline = vi.getTimerCount();

    const check = validate();
    // Nothing is armed in the turn that sends the check, so code stepping
    // timers right after rendering the app meets no deadline...
    expect(vi.getTimerCount()).toBe(baseline);
    await vi.advanceTimersByTimeAsync(0);
    // ...it runs while the check waits...
    expect(vi.getTimerCount()).toBe(baseline + 1);
    answer({ user: admin, isValid: true });
    await check;

    // ...and nothing is left once the answer is in.
    expect(vi.getTimerCount()).toBe(baseline);
  });
});

describe("telling Sentry about session checks that never get an answer", () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("warns once per page load for each kind of check, and not on the console", async () => {
    // A fresh page load: what was reported is remembered per load.
    vi.resetModules();
    const helpers = await import("../helpers");
    const warn = vi.spyOn(console, "warn");
    const error = vi.spyOn(console, "error");
    const never = () => new Promise<never>(() => {});
    actions.validateSession
      .mockImplementationOnce(never)
      .mockImplementationOnce(never);
    actions.getCurrentUser.mockImplementationOnce(never);
    actions.refreshSession.mockImplementationOnce(never);

    const checks = Promise.all([
      helpers.validateSession({ path: "/library", currentUser: admin }),
      helpers.validateSession({ path: "/library", currentUser: admin }),
      helpers.fetchUser(),
      helpers.refreshSession(),
    ]);
    await vi.advanceTimersByTimeAsync(SESSION_CHECK_DEADLINE_MS);
    await checks;

    expect(sentry.addBreadcrumb).toHaveBeenCalledTimes(4);
    expect(sentry.captureMessage).toHaveBeenCalledTimes(3);
    for (const check of [
      "validateSession",
      "getCurrentUser",
      "refreshSession",
    ]) {
      expect(sentry.captureMessage).toHaveBeenCalledWith(
        "Session check got no answer in time",
        {
          level: "warning",
          tags: { context: "session-check", server_action: check },
        },
      );
    }
    // Sentry captures console warnings and errors as well.
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
  });
});
