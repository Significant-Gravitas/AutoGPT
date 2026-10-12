import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { usePlatformChrome } from "../usePlatformChrome";

const pathnameMock = vi.fn<() => string>(() => "/marketplace");
vi.mock("next/navigation", () => ({
  usePathname: () => pathnameMock(),
}));

const authMock = vi.fn(() => ({
  isLoggedIn: true,
  isUserLoading: false,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => authMock(),
}));

describe("usePlatformChrome", () => {
  beforeEach(() => {
    pathnameMock.mockReturnValue("/marketplace");
    authMock.mockReturnValue({ isLoggedIn: true, isUserLoading: false });
  });

  it("shows the app sidebar from the very first render", () => {
    const { result } = renderHook(() => usePlatformChrome());

    // What the server paints: no mount gate, no flag to wait for.
    expect(result.current.showAppSidebar).toBe(true);
  });

  it.each(["/settings", "/settings/billing", "/admin/marketplace"])(
    "keeps the app sidebar off %s, which brings its own shell",
    async (route) => {
      pathnameMock.mockReturnValue(route);
      const { result } = renderHook(() => usePlatformChrome());

      await waitFor(() => {
        expect(result.current.showAppSidebar).toBe(false);
      });
    },
  );

  it.each([
    "/reset-password",
    "/auth/auth-code-error",
    "/error",
    "/unauthorized",
  ])(
    "keeps the app sidebar off the unauthenticated %s route",
    async (route) => {
      pathnameMock.mockReturnValue(route);
      const { result } = renderHook(() => usePlatformChrome());

      await waitFor(() => {
        expect(result.current.showAppSidebar).toBe(false);
      });
    },
  );

  it("collapses the sidebar by default on the builder", () => {
    pathnameMock.mockReturnValue("/build");
    const { result } = renderHook(() => usePlatformChrome());

    expect(result.current.isBuilderRoute).toBe(true);
    expect(result.current.showAppSidebar).toBe(true);
  });

  it("shows the tour sidebar for logged-out marketplace visitors", async () => {
    authMock.mockReturnValue({ isLoggedIn: false, isUserLoading: false });
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(true);
    });
    expect(result.current.showAppSidebar).toBe(false);
  });

  it("keeps the tour sidebar hidden while the session check is in flight", async () => {
    authMock.mockReturnValue({ isLoggedIn: false, isUserLoading: true });
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(false);
    });
    expect(result.current.showAppSidebar).toBe(true);
  });

  it("keeps the tour sidebar hidden for logged-in marketplace visitors", async () => {
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(false);
      expect(result.current.showAppSidebar).toBe(true);
    });
  });

  it("keeps the tour sidebar off non-marketplace routes when logged out", async () => {
    authMock.mockReturnValue({ isLoggedIn: false, isUserLoading: false });
    pathnameMock.mockReturnValue("/library");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(false);
    });
    expect(result.current.showAppSidebar).toBe(true);
  });

  it("gives home the chat controls and floating header", () => {
    pathnameMock.mockReturnValue("/home");
    const { result } = renderHook(() => usePlatformChrome());

    expect(result.current.isCopilotRoute).toBe(true);
    expect(result.current.overlayInsetHeader).toBe(true);
  });
});
