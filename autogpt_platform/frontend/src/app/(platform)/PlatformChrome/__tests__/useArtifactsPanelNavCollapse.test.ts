import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useArtifactsPanelNavCollapse } from "../components/ArtifactsPanelNavCollapse/useArtifactsPanelNavCollapse";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";

const mockUsePathname = vi.fn<() => string | null>();
const mockSetOpen = vi.fn<(open: boolean) => void>();
let sidebarOpen = true;

vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
}));

vi.mock("@/components/ui/sidebar", () => ({
  useSidebar: () => ({ open: sidebarOpen, setOpen: mockSetOpen }),
}));

/** The docked panel — an open artifact preview, or the artifacts tab. */
function setPanelOpen(isOpen: boolean) {
  useCopilotUIStore.setState((s) => ({
    artifactPanel: {
      ...s.artifactPanel,
      isOpen,
      activeTab: isOpen ? "artifacts" : s.artifactPanel.activeTab,
      activeArtifact: null,
    },
  }));
}

function renderAt(pathname: string) {
  mockUsePathname.mockReturnValue(pathname);
  return renderHook(() => useArtifactsPanelNavCollapse());
}

beforeEach(() => {
  mockUsePathname.mockReset();
  mockSetOpen.mockReset();
  sidebarOpen = true;
  setPanelOpen(false);
});

describe("useArtifactsPanelNavCollapse", () => {
  it("collapses the nav when the artifacts panel opens", () => {
    const { rerender } = renderAt("/copilot");
    expect(mockSetOpen).not.toHaveBeenCalled();

    setPanelOpen(true);
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(false);
  });

  it("restores the nav when the panel closes", () => {
    const { rerender } = renderAt("/copilot");
    setPanelOpen(true);
    rerender();
    sidebarOpen = false;

    mockSetOpen.mockClear();
    setPanelOpen(false);
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(true);
  });

  it("leaves a nav that was already collapsed before the panel opened", () => {
    sidebarOpen = false;
    const { rerender } = renderAt("/copilot");
    setPanelOpen(true);
    rerender();

    mockSetOpen.mockClear();
    setPanelOpen(false);
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(false);
  });

  it("keeps a nav the user re-opened by hand while the panel was up", () => {
    // Starts collapsed, so the pre-panel state to restore is "closed" — the
    // case where a blind restore visibly snaps the nav shut on the user.
    sidebarOpen = false;
    const { rerender } = renderAt("/copilot");
    setPanelOpen(true);
    rerender();

    sidebarOpen = true;
    mockSetOpen.mockClear();
    setPanelOpen(false);
    rerender();

    expect(mockSetOpen).not.toHaveBeenCalled();
  });

  it("restores the nav when leaving /copilot with the panel still open", () => {
    const { rerender } = renderAt("/copilot");
    setPanelOpen(true);
    rerender();
    sidebarOpen = false;

    mockSetOpen.mockClear();
    mockUsePathname.mockReturnValue("/library");
    rerender();

    expect(mockSetOpen).toHaveBeenCalledWith(true);
  });

  it("ignores an artifacts panel left open on another route", () => {
    setPanelOpen(true);
    renderAt("/library");

    expect(mockSetOpen).not.toHaveBeenCalled();
  });
});
