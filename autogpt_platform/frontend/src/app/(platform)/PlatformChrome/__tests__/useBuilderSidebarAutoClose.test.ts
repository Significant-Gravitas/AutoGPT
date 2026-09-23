import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useBuilderSidebarAutoClose } from "../components/BuilderSidebarAutoClose/useBuilderSidebarAutoClose";

const mockUsePathname = vi.fn<() => string | null>();
const mockSetOpen = vi.fn<(open: boolean) => void>();
let sidebarOpen = true;

vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
}));

vi.mock("@/components/ui/sidebar", () => ({
  useSidebar: () => ({ open: sidebarOpen, setOpen: mockSetOpen }),
}));

function renderAt(pathname: string) {
  mockUsePathname.mockReturnValue(pathname);
  return renderHook(() => useBuilderSidebarAutoClose());
}

beforeEach(() => {
  mockUsePathname.mockReset();
  mockSetOpen.mockReset();
  sidebarOpen = true;
});

describe("useBuilderSidebarAutoClose", () => {
  it("closes the sidebar when navigating into the builder", () => {
    const { rerender } = renderAt("/marketplace");
    expect(mockSetOpen).not.toHaveBeenCalled();

    mockUsePathname.mockReturnValue("/build");
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(false);
  });

  it("restores the pre-builder state when navigating away", () => {
    const { rerender } = renderAt("/marketplace");

    mockUsePathname.mockReturnValue("/build/123");
    rerender();
    sidebarOpen = false;

    mockSetOpen.mockClear();
    mockUsePathname.mockReturnValue("/library");
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(true);
  });

  it("leaves a sidebar that was already closed before the builder closed", () => {
    sidebarOpen = false;
    const { rerender } = renderAt("/marketplace");

    mockUsePathname.mockReturnValue("/build");
    rerender();

    mockSetOpen.mockClear();
    mockUsePathname.mockReturnValue("/marketplace");
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(false);
  });

  it("does not re-close the sidebar when it is manually re-opened on /build", () => {
    const { rerender } = renderAt("/build");
    expect(mockSetOpen).toHaveBeenCalledWith(false);

    mockSetOpen.mockClear();
    sidebarOpen = true;
    rerender();

    expect(mockSetOpen).not.toHaveBeenCalled();
  });

  it("restores the app default after a hard load straight into the builder", () => {
    // A hard load seeds the provider closed (SidebarProvider defaultOpen), so
    // the seeded `false` must not be mistaken for a user preference.
    sidebarOpen = false;
    const { rerender } = renderAt("/build");

    mockSetOpen.mockClear();
    mockUsePathname.mockReturnValue("/marketplace");
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(true);
  });

  it("ignores repeated navigation between builder sub-routes", () => {
    const { rerender } = renderAt("/build");
    mockSetOpen.mockClear();

    mockUsePathname.mockReturnValue("/build/abc");
    rerender();

    expect(mockSetOpen).not.toHaveBeenCalled();
  });
});
