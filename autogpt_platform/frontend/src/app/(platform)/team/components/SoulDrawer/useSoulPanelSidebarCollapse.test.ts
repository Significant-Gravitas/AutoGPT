import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSoulPanelSidebarCollapse } from "./useSoulPanelSidebarCollapse";

const setOpen = vi.fn<(open: boolean) => void>();
const setOpenMobile = vi.fn<(open: boolean) => void>();
let sidebarOpen = true;

vi.mock("@/components/ui/sidebar", () => ({
  useOptionalSidebar: () => ({
    open: sidebarOpen,
    setOpen,
    setOpenMobile,
  }),
}));

beforeEach(() => {
  sidebarOpen = true;
  setOpen.mockReset();
  setOpenMobile.mockReset();
});

describe("useSoulPanelSidebarCollapse", () => {
  it("closes the left sidebar when the Soul panel opens", () => {
    renderHook(() => useSoulPanelSidebarCollapse(true));

    expect(setOpen).toHaveBeenCalledWith(false);
    expect(setOpenMobile).toHaveBeenCalledWith(false);
  });

  it("restores the previous sidebar state when the Soul panel closes", () => {
    const { rerender } = renderHook(
      ({ isOpen }) => useSoulPanelSidebarCollapse(isOpen),
      { initialProps: { isOpen: true } },
    );
    sidebarOpen = false;
    setOpen.mockClear();

    rerender({ isOpen: false });

    expect(setOpen).toHaveBeenCalledWith(true);
  });
  it("restores a sidebar it closed when the panel unmounts", () => {
    const { rerender, unmount } = renderHook(() =>
      useSoulPanelSidebarCollapse(true),
    );
    sidebarOpen = false;
    rerender();
    setOpen.mockClear();
    unmount();
    expect(setOpen).toHaveBeenCalledExactlyOnceWith(true);
  });

  it("does not open a sidebar that was already closed", () => {
    sidebarOpen = false;
    const { unmount } = renderHook(() => useSoulPanelSidebarCollapse(true));
    setOpen.mockClear();
    unmount();
    expect(setOpen).not.toHaveBeenCalled();
  });

  it("does not overwrite a sidebar the user reopened", () => {
    const { rerender, unmount } = renderHook(() =>
      useSoulPanelSidebarCollapse(true),
    );
    sidebarOpen = false;
    rerender();
    sidebarOpen = true;
    rerender();
    setOpen.mockClear();
    unmount();
    expect(setOpen).not.toHaveBeenCalled();
  });
});
