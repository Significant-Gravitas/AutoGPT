import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useBuilderSidebarAutoClose } from "../components/BuilderSidebarAutoClose/useBuilderSidebarAutoClose";

const mockUsePathname = vi.fn<() => string | null>();
const mockSetOpen =
  vi.fn<(open: boolean, options?: { persist?: boolean }) => void>();
let sidebarOpen = true;

vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
}));

vi.mock("@/components/ui/sidebar", () => ({
  useSidebar: () => ({ open: sidebarOpen, setOpen: mockSetOpen }),
  getPersistedSidebarOpen: () => undefined,
}));

function renderAt(pathname: string, openOutsideBuilder = true) {
  mockUsePathname.mockReturnValue(pathname);
  return renderHook(() => useBuilderSidebarAutoClose(openOutsideBuilder));
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

    expect(mockSetOpen).toHaveBeenCalledWith(false, { persist: false });
  });

  it("restores the pre-builder state when navigating away", () => {
    const { rerender } = renderAt("/marketplace");

    mockUsePathname.mockReturnValue("/build/123");
    rerender();
    sidebarOpen = false;

    mockSetOpen.mockClear();
    mockUsePathname.mockReturnValue("/library");
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(true, { persist: false });
  });

  it("leaves a sidebar that was already closed before the builder closed", () => {
    sidebarOpen = false;
    const { rerender } = renderAt("/marketplace");

    mockUsePathname.mockReturnValue("/build");
    rerender();

    mockSetOpen.mockClear();
    mockUsePathname.mockReturnValue("/marketplace");
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(false, { persist: false });
  });

  it("does not re-close the sidebar when it is manually re-opened on /build", () => {
    const { rerender } = renderAt("/build");
    expect(mockSetOpen).toHaveBeenCalledWith(false, { persist: false });

    mockSetOpen.mockClear();
    sidebarOpen = true;
    rerender();

    expect(mockSetOpen).not.toHaveBeenCalled();
  });

  it.each([true, false])(
    "restores what other routes open with (%s) after a hard load straight into the builder",
    (openOutsideBuilder) => {
      // A hard load seeds the provider closed (SidebarProvider defaultOpen), so
      // the seeded `false` must not be mistaken for the user's choice.
      sidebarOpen = false;
      const { rerender } = renderAt("/build", openOutsideBuilder);

      mockSetOpen.mockClear();
      mockUsePathname.mockReturnValue("/marketplace");
      rerender();

      expect(mockSetOpen).toHaveBeenCalledWith(openOutsideBuilder, {
        persist: false,
      });
    },
  );

  it("ignores repeated navigation between builder sub-routes", () => {
    const { rerender } = renderAt("/build");
    mockSetOpen.mockClear();

    mockUsePathname.mockReturnValue("/build/abc");
    rerender();

    expect(mockSetOpen).not.toHaveBeenCalled();
  });
});
