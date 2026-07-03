import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PlatformChrome } from "../PlatformChrome";

const showNewLayoutMock = vi.fn<() => boolean>(() => false);
vi.mock("../usePlatformChrome", () => ({
  usePlatformChrome: () => ({ showNewLayout: showNewLayoutMock() }),
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
});
