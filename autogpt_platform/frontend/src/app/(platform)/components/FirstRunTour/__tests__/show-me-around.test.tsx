import {
  AppSidebarToggleProvider,
  AppSidebarTrigger,
} from "@/components/layout/AppSidebar/components/AppSidebarTrigger/AppSidebarTrigger";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { act, render, renderHook } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const sidebar = vi.hoisted(() => ({
  current: { isMobile: true, setOpenMobile: vi.fn() } as {
    isMobile: boolean;
    setOpenMobile: (open: boolean) => void;
  } | null,
}));
vi.mock("@/components/molecules/Sidebar/Sidebar", async (original) => ({
  ...(await original<object>()),
  useOptionalSidebar: () => sidebar.current,
}));

const nav = vi.hoisted(() => ({ pathname: "/build", push: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: nav.push,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => nav.pathname,
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { useFirstRunTourStore } from "../store";
import { useReplayOnArrival } from "../useReplayOnArrival";
import { useShowMeAround } from "../useShowMeAround";

afterEach(() => {
  act(() => useFirstRunTourStore.getState().close());
});

describe("Show me around", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/home");
    nav.pathname = "/home";
  });
  afterEach(() => {
    window.history.replaceState({}, "", "/");
  });

  it("closes the phone's sidebar sheet before the walkthrough opens over it", () => {
    const setOpenMobile = vi.fn();
    sidebar.current = { isMobile: true, setOpenMobile };
    const { result } = renderHook(() => useShowMeAround());

    act(() => result.current());

    expect(setOpenMobile).toHaveBeenCalledWith(false);
    expect(useFirstRunTourStore.getState().source).toBe("menu");
  });

  it("leaves the desktop sidebar alone", () => {
    const setOpenMobile = vi.fn();
    sidebar.current = { isMobile: false, setOpenMobile };
    const { result } = renderHook(() => useShowMeAround());

    act(() => result.current());

    expect(setOpenMobile).not.toHaveBeenCalled();
    expect(useFirstRunTourStore.getState().source).toBe("menu");
  });
});

describe("Show me around, from another page", () => {
  afterEach(() => {
    window.history.replaceState({}, "", "/");
    nav.pathname = "/build";
    nav.push.mockClear();
  });

  it("opens the walkthrough once Home has loaded, not over the page it left", () => {
    // Opened before the route lands, it mounted over the builder's
    // forced-collapsed sidebar and borrowed (and later gave back) that.
    window.history.replaceState({}, "", "/build");
    nav.pathname = "/build";
    sidebar.current = null;
    const { result, rerender } = renderHook(() => {
      useReplayOnArrival();
      return useShowMeAround();
    });

    act(() => result.current());

    expect(nav.push).toHaveBeenCalledWith("/home");
    expect(useFirstRunTourStore.getState().source).toBeNull();

    nav.pathname = "/home";
    rerender();

    expect(useFirstRunTourStore.getState().source).toBe("menu");
    expect(useFirstRunTourStore.getState().pendingReplay).toBe(false);
  });

  it("opens straight away on Home itself", () => {
    window.history.replaceState({}, "", "/home");
    nav.pathname = "/home";
    sidebar.current = null;
    const { result } = renderHook(() => {
      useReplayOnArrival();
      return useShowMeAround();
    });

    act(() => result.current());

    expect(nav.push).not.toHaveBeenCalled();
    expect(useFirstRunTourStore.getState().source).toBe("menu");
  });
});

describe("the menu button inside a chat", () => {
  it("is what the walkthrough and tips point at on a phone", () => {
    const { getByRole } = render(
      <SidebarProvider>
        <AppSidebarToggleProvider>
          <AppSidebarTrigger />
        </AppSidebarToggleProvider>
      </SidebarProvider>,
    );

    expect(
      getByRole("button", { name: "Toggle Sidebar" }).getAttribute(
        "data-tour-target",
      ),
    ).toBe("menu");
  });
});
