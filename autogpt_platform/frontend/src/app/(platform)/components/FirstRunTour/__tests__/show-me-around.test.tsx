import {
  AppSidebarToggleProvider,
  AppSidebarTrigger,
} from "@/components/layout/AppSidebar/components/AppSidebarTrigger/AppSidebarTrigger";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { act, render, renderHook } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

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

import { useFirstRunTourStore } from "../store";
import { useShowMeAround } from "../useShowMeAround";

afterEach(() => {
  act(() => useFirstRunTourStore.getState().close());
});

describe("Show me around", () => {
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
