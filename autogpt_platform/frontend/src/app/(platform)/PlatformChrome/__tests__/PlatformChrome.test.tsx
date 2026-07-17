import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useTourStore } from "@/app/(public)/tour/chat/tourStore";
import { PlatformChrome } from "../PlatformChrome";

const showNewLayoutMock = vi.fn<() => boolean>(() => false);
const showTourSidebarMock = vi.fn<() => boolean>(() => false);
vi.mock("../usePlatformChrome", () => ({
  usePlatformChrome: () => ({
    showNewLayout: showNewLayoutMock(),
    showTourSidebar: showTourSidebarMock(),
  }),
}));

// Mirrors the global setup-nextjs-mocks shape, but with a stable push spy so
// the tour sidebar's navigation can be asserted.
const { pushMock } = vi.hoisted(() => ({ pushMock: vi.fn() }));
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/components/layout/AppSidebar/AppSidebar", () => ({
  AppSidebar: () => <div data-testid="app-sidebar" />,
}));
vi.mock("@/components/layout/Navbar/Navbar", () => ({
  Navbar: () => <div data-testid="navbar" />,
}));
vi.mock("@/components/layout/TopUpPrompt/TopUpPromptProvider", () => ({
  TopUpPromptProvider: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
}));
vi.mock("../../PaywallGate/PaywallGate", () => ({
  PaywallGate: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));
vi.mock("../../admin/components/AdminImpersonationBanner", () => ({
  AdminImpersonationBanner: () => null,
}));
vi.mock("../../components/GlobalSearchModal/GlobalSearchOverlay", () => ({
  GlobalSearchOverlay: () => <div data-testid="global-search" />,
}));
vi.mock("../components/InsetHeaderActions/InsetHeaderActions", () => ({
  InsetHeaderActions: () => <div data-testid="inset-actions" />,
}));

afterEach(() => {
  vi.clearAllMocks();
});

describe("PlatformChrome", () => {
  beforeEach(() => {
    showNewLayoutMock.mockReturnValue(false);
    showTourSidebarMock.mockReturnValue(false);
  });

  it("renders the classic Navbar shell when the new layout is off", () => {
    render(
      <PlatformChrome>
        <div data-testid="child">content</div>
      </PlatformChrome>,
    );

    expect(screen.getByTestId("navbar")).toBeDefined();
    expect(screen.queryByTestId("app-sidebar")).toBeNull();
    expect(screen.getByTestId("child")).toBeDefined();
  });

  it("renders the new sidebar shell with inset actions when enabled", async () => {
    showNewLayoutMock.mockReturnValue(true);
    render(
      <PlatformChrome>
        <div data-testid="child">content</div>
      </PlatformChrome>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("app-sidebar")).toBeDefined();
    });
    expect(screen.getByTestId("inset-actions")).toBeDefined();
    expect(screen.queryByTestId("navbar")).toBeNull();
    expect(screen.getByTestId("child")).toBeDefined();
  });

  it("renders the tour upsell sidebar shell when showTourSidebar is on", async () => {
    showTourSidebarMock.mockReturnValue(true);
    render(
      <PlatformChrome>
        <div data-testid="child">content</div>
      </PlatformChrome>,
    );

    expect(screen.getByText("Try Autopilot")).toBeDefined();
    expect(screen.getByText(/Ready to build your own/i)).toBeDefined();
    expect(screen.queryByTestId("navbar")).toBeNull();
    expect(screen.queryByTestId("app-sidebar")).toBeNull();
    expect(screen.getByTestId("child")).toBeDefined();

    // Clicking a demo session stores the scenario and navigates to the tour.
    fireEvent.click(screen.getByRole("button", { name: "Daily brief" }));
    expect(useTourStore.getState().activeScenarioId).toBe("daily-brief");
    expect(pushMock).toHaveBeenCalledWith(
      "/tour/chat?utm_source=platform_marketplace",
    );
  });
});
