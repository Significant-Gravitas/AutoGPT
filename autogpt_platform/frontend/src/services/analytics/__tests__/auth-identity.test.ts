import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  captureFirstLanding,
  getAnonymousID,
  readFirstLanding,
  resetAnonymousIDForTests,
} from "../anonymous-id";

const posthog = vi.hoisted(() => ({
  get_distinct_id: vi.fn<() => string | undefined>(),
  reset: vi.fn(),
}));

vi.mock("posthog-js", () => ({ default: posthog }));
vi.mock("@/lib/auth/actions", () => ({ serverLogout: vi.fn() }));
vi.mock("@/lib/auth/helpers", () => ({
  broadcastLogout: vi.fn(),
  setWebSocketDisconnectIntent: vi.fn(),
  setupSessionEventListeners: vi.fn(() => ({ cleanup: vi.fn() })),
}));
vi.mock("@/lib/auth/hooks/helpers", () => ({
  fetchUser: vi.fn(),
  handleStorageEvent: vi.fn(),
  refreshSession: vi.fn(),
  validateSession: vi.fn(),
}));

const userA = { id: "user-A", email: "a@example.com" } as User;
const userB = { id: "user-B", email: "b@example.com" } as User;

beforeEach(() => {
  useAuthStore.setState({ user: null });
  window.localStorage.clear();
  resetAnonymousIDForTests();
  vi.resetAllMocks();
  vi.stubEnv("NEXT_PUBLIC_POSTHOG_KEY", "phc_test");
});

afterEach(() => {
  useAuthStore.setState({ user: null });
  vi.unstubAllEnvs();
});

describe("analytics identity on account transitions", () => {
  it("rotates the visitor and landing on logout without an initialized PostHog client", () => {
    window.localStorage.setItem(
      "ph_phc_test_posthog",
      JSON.stringify({ $device_id: "old-device" }),
    );
    expect(getAnonymousID()).toBe("old-device");
    captureFirstLanding();
    useAuthStore.setState({ user: userA });

    useAuthStore.setState({ user: null });

    expect(getAnonymousID()).not.toBe("old-device");
    expect(readFirstLanding()).toBeNull();
    expect(posthog.reset).not.toHaveBeenCalled();
  });

  it.each([null, userB])(
    "shares the fresh PostHog identity before observers see the next account %j",
    (nextUser) => {
      posthog.get_distinct_id.mockReturnValue("user-A");
      posthog.reset.mockImplementation(() => {
        posthog.get_distinct_id.mockReturnValue("new-visitor");
      });
      useAuthStore.setState({ user: userA });
      const observed: Array<string | null> = [];
      const unsubscribe = useAuthStore.subscribe(() => {
        observed.push(getAnonymousID());
      });

      try {
        useAuthStore.setState({ user: nextUser });
      } finally {
        unsubscribe();
      }

      expect(posthog.reset).toHaveBeenCalledOnce();
      expect(posthog.reset).toHaveBeenCalledWith(true);
      expect(observed).toEqual(["new-visitor"]);
      expect(getAnonymousID()).toBe(posthog.get_distinct_id());
    },
  );

  it("keeps account changes working when the analytics client throws", () => {
    useAuthStore.setState({ user: userA });
    const previous = getAnonymousID();
    posthog.get_distinct_id.mockImplementation(() => {
      throw new Error("Analytics storage unavailable");
    });

    expect(() => useAuthStore.setState({ user: null })).not.toThrow();

    expect(getAnonymousID()).not.toBe(previous);
    expect(useAuthStore.getState().user).toBeNull();
  });

  it("preserves the anonymous identity through login and same-user refreshes", () => {
    const previous = getAnonymousID();
    useAuthStore.setState({ user: userA });
    useAuthStore.setState({ user: { ...userA } });

    expect(getAnonymousID()).toBe(previous);
    expect(posthog.reset).not.toHaveBeenCalled();
  });
});
