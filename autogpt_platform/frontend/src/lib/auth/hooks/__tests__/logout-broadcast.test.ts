import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Only the server actions are faked. The store and the real cross-tab
// broadcast (a localStorage write other tabs listen for) run as shipped.
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

import { useAuthStore } from "../useAuthStore";

const LOGOUT_SIGNAL = "supabase-logout";

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  useAuthStore.getState().cleanup();
  vi.clearAllMocks();
});

describe("signing out announces it to other tabs", () => {
  it("only once the server has ended the session, however slow that is", async () => {
    let finishServerLogout!: () => void;
    actions.serverLogout.mockReturnValue(
      new Promise<{ success: boolean }>((resolve) => {
        finishServerLogout = () => resolve({ success: true });
      }),
    );

    const loggingOut = useAuthStore.getState().logOut();
    await vi.waitFor(() => expect(actions.serverLogout).toHaveBeenCalled());

    // Another tab that reloaded now would still find a live session cookie
    // and land back in the app, so it must not have been told yet.
    expect(localStorage.getItem(LOGOUT_SIGNAL)).toBeNull();

    finishServerLogout();
    await loggingOut;

    expect(localStorage.getItem(LOGOUT_SIGNAL)).not.toBeNull();
  });

  it("even when the server sign-out fails, so other tabs don't stay signed in", async () => {
    actions.serverLogout.mockRejectedValue(new Error("network down"));

    await expect(useAuthStore.getState().logOut()).rejects.toThrow(
      "network down",
    );

    expect(localStorage.getItem(LOGOUT_SIGNAL)).not.toBeNull();
  });
});
