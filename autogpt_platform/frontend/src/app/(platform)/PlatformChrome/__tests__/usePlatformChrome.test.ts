import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { usePlatformChrome } from "../usePlatformChrome";

const pathnameMock = vi.fn<() => string>(() => "/marketplace");
vi.mock("next/navigation", () => ({
  usePathname: () => pathnameMock(),
}));

const supabaseMock = vi.fn(() => ({
  isLoggedIn: true,
  isUserLoading: false,
}));
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => supabaseMock(),
}));

const flagMock = vi.fn<(flag: string) => boolean>(() => true);
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flagMock(flag),
  };
});

describe("usePlatformChrome", () => {
  beforeEach(() => {
    pathnameMock.mockReturnValue("/marketplace");
    flagMock.mockReturnValue(true);
    supabaseMock.mockReturnValue({ isLoggedIn: true, isUserLoading: false });
  });

  it("enables the new layout after mount when the flag is on and route is allowed", async () => {
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(true);
    });
  });

  it("keeps the classic layout when the flag is off", async () => {
    flagMock.mockReturnValue(false);
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(false);
    });
  });

  it("excludes the /settings route from the new layout", async () => {
    pathnameMock.mockReturnValue("/settings");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      // give the mount effect a chance to run; it should still be false.
      expect(result.current.showNewLayout).toBe(false);
    });
  });

  it("excludes nested /settings/* routes from the new layout", async () => {
    pathnameMock.mockReturnValue("/settings/billing");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(false);
    });
  });

  it.each([
    "/reset-password",
    "/auth/auth-code-error",
    "/error",
    "/unauthorized",
  ])(
    "excludes the unauthenticated %s route from the new layout",
    async (route) => {
      pathnameMock.mockReturnValue(route);
      const { result } = renderHook(() => usePlatformChrome());

      await waitFor(() => {
        expect(result.current.showNewLayout).toBe(false);
      });
    },
  );

  it("passes the flag enum to useGetFlag", async () => {
    renderHook(() => usePlatformChrome());
    await waitFor(() => {
      expect(flagMock).toHaveBeenCalledWith("autogpt-new-layout");
    });
  });

  it("shows the tour sidebar for logged-out marketplace visitors", async () => {
    supabaseMock.mockReturnValue({ isLoggedIn: false, isUserLoading: false });
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(true);
    });
    expect(result.current.showNewLayout).toBe(false);
  });

  it("keeps the tour sidebar hidden while the session check is in flight", async () => {
    supabaseMock.mockReturnValue({ isLoggedIn: false, isUserLoading: true });
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(false);
    });
  });

  it("keeps the tour sidebar hidden for logged-in marketplace visitors", async () => {
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(false);
      expect(result.current.showNewLayout).toBe(true);
    });
  });

  it("keeps the tour sidebar off non-marketplace routes when logged out", async () => {
    supabaseMock.mockReturnValue({ isLoggedIn: false, isUserLoading: false });
    pathnameMock.mockReturnValue("/library");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(false);
    });
  });
});
