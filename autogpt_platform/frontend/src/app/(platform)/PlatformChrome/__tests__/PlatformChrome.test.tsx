import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useTourStore } from "@/app/(public)/tour/chat/tourStore";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import { PlatformChrome } from "../PlatformChrome";

const showAppSidebarMock = vi.fn<() => boolean>(() => true);
const showTourSidebarMock = vi.fn<() => boolean>(() => false);
const showChangelogMock = vi.fn<() => boolean>(() => true);
vi.mock("../usePlatformChrome", () => ({
  usePlatformChrome: () => ({
    showAppSidebar: showAppSidebarMock(),
    showTourSidebar: showTourSidebarMock(),
    showChangelog: showChangelogMock(),
  }),
}));

vi.mock("@/components/layout/AppSidebar/AppSidebar", () => ({
  AppSidebar: () => <div data-testid="app-sidebar" />,
}));
vi.mock("@/components/molecules/ChangelogPopup/ChangelogPopup", () => ({
  ChangelogPopup: () => <div data-testid="changelog-popup" />,
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

afterEach(() => {
  vi.clearAllMocks();
  removeCookiebot();
  vi.unstubAllEnvs();
});

describe("PlatformChrome", () => {
  beforeEach(() => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    showAppSidebarMock.mockReturnValue(true);
    showTourSidebarMock.mockReturnValue(false);
    showChangelogMock.mockReturnValue(true);
    useTourStore.setState({ isDemoComplete: false });
  });

  it("renders the app sidebar shell", () => {
    render(
      <PlatformChrome>
        <div data-testid="child">content</div>
      </PlatformChrome>,
    );

    expect(screen.getByTestId("app-sidebar")).toBeDefined();
    expect(screen.getByTestId("changelog-popup")).toBeDefined();
    expect(screen.getByTestId("global-search")).toBeDefined();
    expect(screen.getByTestId("child")).toBeDefined();
  });

  it("renders a bare frame with no sidebar on excluded routes", () => {
    showAppSidebarMock.mockReturnValue(false);
    render(
      <PlatformChrome>
        <div data-testid="child">content</div>
      </PlatformChrome>,
    );

    expect(screen.queryByTestId("app-sidebar")).toBeNull();
    expect(screen.queryByTestId("changelog-popup")).toBeNull();
    expect(screen.queryByRole("navigation")).toBeNull();
    expect(screen.getByTestId("global-search")).toBeDefined();
    expect(screen.getByTestId("child")).toBeDefined();
  });

  it("renders a free-trial sidebar without demo chats when logged out", () => {
    showAppSidebarMock.mockReturnValue(false);
    showTourSidebarMock.mockReturnValue(true);
    render(
      <PlatformChrome>
        <div data-testid="child">content</div>
      </PlatformChrome>,
    );

    expect(screen.queryByText("Try Otto")).toBeNull();
    expect(screen.queryByText("Recent chats")).toBeNull();
    for (const label of [
      "Daily brief",
      "Call prep",
      "Competitor watch",
      "Support queue",
    ]) {
      expect(screen.queryByRole("button", { name: label })).toBeNull();
    }
    expect(screen.getByText("Your AI team starts here")).toBeDefined();
    const trialCTA = screen.getByRole("link", { name: "Start free trial" });
    expect(trialCTA.getAttribute("href")).toBe("/signup");
    expect(trialCTA.getAttribute("target")).toBeNull();
    expect(
      screen.getByRole("link", { name: "AutoGPT" }).getAttribute("href"),
    ).toBe("/marketplace");
    expect(screen.queryByText(/Start with Pro/i)).toBeNull();
    expect(screen.queryByText(/\$42\.50/)).toBeNull();
    expect(screen.queryByTestId("app-sidebar")).toBeNull();
    expect(screen.queryByTestId("changelog-popup")).toBeNull();
    expect(screen.getByTestId("child")).toBeDefined();
  });

  it("waits for the signed-in session before showing the changelog", () => {
    showChangelogMock.mockReturnValue(false);
    render(<PlatformChrome>content</PlatformChrome>);

    expect(screen.getByTestId("app-sidebar")).toBeDefined();
    expect(screen.queryByTestId("changelog-popup")).toBeNull();
  });

  it("keeps the marketplace free-trial card after a previous tour completes", () => {
    showTourSidebarMock.mockReturnValue(true);
    useTourStore.setState({ isDemoComplete: true });

    render(
      <PlatformChrome>
        <div>Marketplace content</div>
      </PlatformChrome>,
    );

    expect(
      screen
        .getByRole("link", { name: "Start free trial" })
        .getAttribute("href"),
    ).toBe("/signup");
  });

  it("offers cookie settings to signed-out visitors in the tour sidebar shell", async () => {
    configureCookiebot();
    const { renew } = installCookiebot();
    showTourSidebarMock.mockReturnValue(true);
    render(
      <PlatformChrome>
        <div data-testid="child">content</div>
      </PlatformChrome>,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: "Cookie settings" }),
    );

    expect(renew).toHaveBeenCalledOnce();
  });
});
