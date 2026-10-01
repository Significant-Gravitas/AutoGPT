import { renderHook, waitFor } from "@testing-library/react";
import { createElement, ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { LayoutHintProvider } from "../components/LayoutHintProvider/LayoutHintProvider";
import { LAYOUT_HINT_COOKIE, LayoutHint } from "../helpers";
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

interface FlagStatus {
  enabled: boolean;
  ready: boolean;
  answered: boolean;
}

const flagMock = vi.fn<(flag: string) => FlagStatus>(() => ({
  enabled: true,
  ready: true,
  answered: true,
}));
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) => flagMock(flag),
  };
});

function setFlag(status: Partial<FlagStatus>) {
  flagMock.mockReturnValue({
    enabled: true,
    ready: true,
    answered: true,
    ...status,
  });
}

function withHint(hint: LayoutHint | undefined) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(LayoutHintProvider, { hint, children });
  };
}

function layoutCookie() {
  return document.cookie
    .split("; ")
    .find((entry) => entry.startsWith(`${LAYOUT_HINT_COOKIE}=`));
}

describe("usePlatformChrome", () => {
  beforeEach(() => {
    pathnameMock.mockReturnValue("/marketplace");
    setFlag({ enabled: true });
    authMock.mockReturnValue({ isLoggedIn: true, isUserLoading: false });
  });

  afterEach(() => {
    document.cookie = `${LAYOUT_HINT_COOKIE}=; path=/; max-age=0`;
  });

  it("enables the new layout after mount when the flag is on and route is allowed", async () => {
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(true);
    });
  });

  it("keeps the classic layout when the flag is off", async () => {
    setFlag({ enabled: false });
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(false);
      expect(result.current.isLayoutPending).toBe(false);
    });
  });

  it("never paints the classic shell before the flag has answered", async () => {
    setFlag({ ready: false, answered: false });
    const { result } = renderHook(() => usePlatformChrome());

    // Same on the server and across mount: no cookie, no answer, no shell.
    expect(result.current.isLayoutPending).toBe(true);
    expect(result.current.showNewLayout).toBe(false);
    expect(result.current.isNewLayoutActive).toBe(false);

    await waitFor(() => {
      expect(result.current.isLayoutPending).toBe(true);
    });
    expect(layoutCookie()).toBeUndefined();
  });

  it("renders the shell the cookie remembers while the flag is still answering", () => {
    setFlag({ ready: false, answered: false });
    const { result } = renderHook(() => usePlatformChrome(), {
      wrapper: withHint("new"),
    });

    expect(result.current.isLayoutPending).toBe(false);
    expect(result.current.showNewLayout).toBe(true);
    expect(result.current.isNewLayoutActive).toBe(true);
  });

  it("honours a classic cookie before the flag has answered", () => {
    setFlag({ ready: false, answered: false });
    const { result } = renderHook(() => usePlatformChrome(), {
      wrapper: withHint("classic"),
    });

    expect(result.current.isLayoutPending).toBe(false);
    expect(result.current.showNewLayout).toBe(false);
  });

  it("lets the flag's answer override a stale cookie", async () => {
    setFlag({ enabled: false });
    const { result } = renderHook(() => usePlatformChrome(), {
      wrapper: withHint("new"),
    });

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(false);
    });
    expect(result.current.isLayoutPending).toBe(false);
  });

  it("remembers the flag's answer in the layout cookie", async () => {
    renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(layoutCookie()).toBe(`${LAYOUT_HINT_COOKIE}=new`);
    });

    setFlag({ enabled: false });
    renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(layoutCookie()).toBe(`${LAYOUT_HINT_COOKIE}=classic`);
    });
  });

  it("keeps the remembered shell when the flag vendor times out", async () => {
    setFlag({ enabled: false, ready: true, answered: false });
    const { result } = renderHook(() => usePlatformChrome(), {
      wrapper: withHint("new"),
    });

    await waitFor(() => {
      expect(result.current.showNewLayout).toBe(true);
    });
    // A timeout is not an answer worth remembering.
    expect(layoutCookie()).toBeUndefined();
  });

  it("falls back to the flag default on a vendor timeout with no cookie", async () => {
    setFlag({ enabled: false, ready: true, answered: false });
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.isLayoutPending).toBe(false);
    });
    expect(result.current.showNewLayout).toBe(false);
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

  it("excludes /admin routes from the new layout but keeps the flag active", async () => {
    pathnameMock.mockReturnValue("/admin/marketplace");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      // The admin section brings its own sidebar, so the app sidebar shell is
      // suppressed even though the new-layout flag itself is active.
      expect(result.current.showNewLayout).toBe(false);
      expect(result.current.isNewLayoutActive).toBe(true);
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

  it("passes the flag enum to useFlagStatus", async () => {
    renderHook(() => usePlatformChrome());
    await waitFor(() => {
      expect(flagMock).toHaveBeenCalledWith("autogpt-new-layout");
    });
  });

  it("shows the tour sidebar for logged-out marketplace visitors", async () => {
    authMock.mockReturnValue({ isLoggedIn: false, isUserLoading: false });
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(true);
    });
    expect(result.current.showNewLayout).toBe(false);
  });

  it("keeps the tour sidebar hidden while the session check is in flight", async () => {
    authMock.mockReturnValue({ isLoggedIn: false, isUserLoading: true });
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
    authMock.mockReturnValue({ isLoggedIn: false, isUserLoading: false });
    pathnameMock.mockReturnValue("/library");
    const { result } = renderHook(() => usePlatformChrome());

    await waitFor(() => {
      expect(result.current.showTourSidebar).toBe(false);
    });
  });
});

it("gives home the chat controls and floating header", () => {
  pathnameMock.mockReturnValue("/home");
  const { result } = renderHook(() => usePlatformChrome());
  expect(result.current.isCopilotRoute).toBe(true);
  expect(result.current.overlayInsetHeader).toBe(true);
});
