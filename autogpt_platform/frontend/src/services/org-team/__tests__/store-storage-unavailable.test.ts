import { beforeEach, describe, expect, it, vi } from "vitest";

const captureException = vi.hoisted(() => vi.fn());
const isServerSide = vi.hoisted(() => vi.fn(() => false));

vi.mock("@sentry/nextjs", () => ({ captureException }));
vi.mock("@/services/environment", () => ({ environment: { isServerSide } }));

/**
 * `OrgTeamProvider` sits in `src/app/providers.tsx`, so every route pulls this
 * store in and zustand runs its initialiser — which reads localStorage — while
 * the module is evaluated. That happens during server rendering too, and it is
 * where the "Local storage is not available" flood came from.
 */
describe("useOrgTeamStore when storage is unavailable", () => {
  beforeEach(() => {
    vi.resetModules();
    captureException.mockClear();
    isServerSide.mockReturnValue(false);
    window.localStorage.clear();
  });

  it("initialises during server rendering without reporting an error", async () => {
    isServerSide.mockReturnValue(true);

    const { useOrgTeamStore } = await import("../store");

    const state = useOrgTeamStore.getState();
    expect(state.activeOrgID).toBeNull();
    expect(state.activeTeamID).toBeNull();
    expect(captureException).not.toHaveBeenCalled();
  });

  it("initialises in a browser that blocks storage without reporting an error", async () => {
    const original = Object.getOwnPropertyDescriptor(window, "localStorage");
    Object.defineProperty(window, "localStorage", {
      get() {
        throw new Error("The operation is insecure.");
      },
      configurable: true,
    });

    try {
      const { useOrgTeamStore } = await import("../store");

      expect(useOrgTeamStore.getState().activeOrgID).toBeNull();

      // Switching org still works, it just does not outlive the tab.
      useOrgTeamStore.getState().setActiveOrg("org-1");
      expect(useOrgTeamStore.getState().activeOrgID).toBe("org-1");

      useOrgTeamStore.getState().clearContext();
      expect(useOrgTeamStore.getState().activeOrgID).toBeNull();
      expect(captureException).not.toHaveBeenCalled();
    } finally {
      if (original) Object.defineProperty(window, "localStorage", original);
    }
  });
});
