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
});
