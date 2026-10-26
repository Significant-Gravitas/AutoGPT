import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { usePanelSidebarCollapse } from "@/app/(platform)/team/components/ExpertSidePanel/usePanelSidebarCollapse";
import { Sidebar, useSidebar } from "@/components/molecules/Sidebar/Sidebar";
import { SIDEBAR_COOKIE_NAME } from "@/components/molecules/Sidebar/sidebar-cookie";
import { PlatformChrome } from "../PlatformChrome";

let pathname = "/home";
let isSoulPanelOpen = false;

vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

vi.mock("../usePlatformChrome", () => ({
  usePlatformChrome: () => ({
    showAppSidebar: true,
    isBuilderRoute: pathname.startsWith("/build"),
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
  };
});

// The real Sidebar, so its rendered open state is what the test reads.
vi.mock("@/components/layout/AppSidebar/AppSidebar", () => ({
  AppSidebar: () => (
    <Sidebar collapsible="icon">
      <nav aria-label="App navigation" />
    </Sidebar>
  ),
}));
vi.mock("@/components/layout/TopUpPrompt/TopUpPromptProvider", () => ({
  TopUpPromptProvider: ({ children }: { children: ReactNode }) => (
    <>{children}</>
  ),
}));
vi.mock("../../PaywallGate/PaywallGate", () => ({
  PaywallGate: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
vi.mock("../../admin/components/AdminImpersonationBanner", () => ({
  AdminImpersonationBanner: () => null,
}));
vi.mock("../../components/GlobalSearchModal/GlobalSearchOverlay", () => ({
  GlobalSearchOverlay: () => null,
}));

const DESKTOP_WIDTH = window.innerWidth;

function load(persistedSidebarOpen?: boolean) {
  const paintedStates: string[] = [];
  function StateLog() {
    paintedStates.push(useSidebar().state);
    return null;
  }
  function SoulPanel() {
    usePanelSidebarCollapse(isSoulPanelOpen);
    return null;
  }
  function tree() {
    return (
      <PlatformChrome persistedSidebarOpen={persistedSidebarOpen}>
        <StateLog />
        <SoulPanel />
      </PlatformChrome>
    );
  }
  const view = render(tree());
  function navigate(to: string) {
    pathname = to;
    view.rerender(tree());
  }
  function setSoulPanelOpen(isOpen: boolean) {
    isSoulPanelOpen = isOpen;
    view.rerender(tree());
  }
  return { ...view, paintedStates, navigate, setSoulPanelOpen };
}

function setArtifactsPanelOpen(isOpen: boolean) {
  act(() =>
    useCopilotUIStore.setState((s) => ({
      artifactPanel: {
        ...s.artifactPanel,
        isOpen,
        activeTab: isOpen ? "artifacts" : s.artifactPanel.activeTab,
        activeArtifact: null,
      },
    })),
  );
}

function sidebarState() {
  return document
    .querySelector("[data-collapsible]")
    ?.getAttribute("data-state");
}

function rememberSidebar(open: boolean) {
  document.cookie = `${SIDEBAR_COOKIE_NAME}=${open}; path=/`;
}

beforeEach(() => {
  pathname = "/home";
  isSoulPanelOpen = false;
  setArtifactsPanelOpen(false);
  document.cookie = `${SIDEBAR_COOKIE_NAME}=; path=/; max-age=0`;
});

afterEach(() => {
  Object.defineProperty(window, "innerWidth", {
    value: DESKTOP_WIDTH,
    configurable: true,
  });
});

describe("the app sidebar across reloads", () => {
  it("starts expanded when nothing is remembered", () => {
    load();

    expect(sidebarState()).toBe("expanded");
  });

  it("keeps a collapse after a reload", () => {
    const first = load();
    fireEvent.click(screen.getByRole("button", { name: "Toggle Sidebar" }));
    expect(sidebarState()).toBe("collapsed");
    first.unmount();

    load();

    expect(sidebarState()).toBe("collapsed");
  });

  it("keeps a re-expand after a reload", () => {
    rememberSidebar(false);
    const first = load();
    fireEvent.click(screen.getByRole("button", { name: "Toggle Sidebar" }));
    expect(sidebarState()).toBe("expanded");
    first.unmount();

    load();

    expect(sidebarState()).toBe("expanded");
  });

  it("falls back to the cookie the server read when the browser has none", () => {
    load(false);

    expect(sidebarState()).toBe("collapsed");
  });

  it("never paints the builder with the sidebar open, whatever was remembered", () => {
    rememberSidebar(true);
    pathname = "/build";

    const { paintedStates } = load();

    expect(paintedStates.length).toBeGreaterThan(0);
    expect(paintedStates).not.toContain("expanded");
    expect(sidebarState()).toBe("collapsed");
  });

  it("does not let the builder's own collapse overwrite the remembered choice", () => {
    rememberSidebar(true);
    pathname = "/build";
    load().unmount();

    pathname = "/home";
    load();

    expect(sidebarState()).toBe("expanded");
  });

  it.each([
    { before: "closed", remembered: false, toggles: 1, expected: "expanded" },
    { before: "open", remembered: true, toggles: 2, expected: "collapsed" },
  ])(
    "leaving the builder after toggling there ($before before, $toggles toggles) shows what a reload would",
    ({ remembered, toggles, expected }) => {
      rememberSidebar(remembered);
      const view = load();
      view.navigate("/build");
      for (let i = 0; i < toggles; i++) {
        fireEvent.click(screen.getByRole("button", { name: "Toggle Sidebar" }));
      }

      view.navigate("/home");

      expect(sidebarState()).toBe(expected);
      view.unmount();
      load();
      expect(sidebarState()).toBe(expected);
    },
  );

  it.each([
    {
      panel: "artifacts",
      before: "closed",
      remembered: false,
      toggles: 1,
      expected: "expanded",
    },
    {
      panel: "artifacts",
      before: "open",
      remembered: true,
      toggles: 2,
      expected: "collapsed",
    },
    {
      panel: "Soul",
      before: "closed",
      remembered: false,
      toggles: 1,
      expected: "expanded",
    },
    {
      panel: "Soul",
      before: "open",
      remembered: true,
      toggles: 2,
      expected: "collapsed",
    },
  ])(
    "closing the $panel panel after toggling under it ($before before, $toggles toggles) shows what a reload would",
    ({ panel, remembered, toggles, expected }) => {
      rememberSidebar(remembered);
      const view = load();
      const setPanelOpen =
        panel === "Soul" ? view.setSoulPanelOpen : setArtifactsPanelOpen;
      setPanelOpen(true);
      expect(sidebarState()).toBe("collapsed");
      for (let i = 0; i < toggles; i++) {
        fireEvent.click(screen.getByRole("button", { name: "Toggle Sidebar" }));
      }

      setPanelOpen(false);

      expect(sidebarState()).toBe(expected);
      view.unmount();
      isSoulPanelOpen = false;
      setArtifactsPanelOpen(false);
      load();
      expect(sidebarState()).toBe(expected);
    },
  );

  it("does not open the mobile sheet over the page when open is remembered", () => {
    Object.defineProperty(window, "innerWidth", {
      value: 375,
      configurable: true,
    });
    rememberSidebar(true);

    load();

    // The desktop sidebar always renders its nav, so its absence proves the mobile path.
    expect(
      screen.queryByRole("navigation", { name: "App navigation" }),
    ).toBeNull();
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
