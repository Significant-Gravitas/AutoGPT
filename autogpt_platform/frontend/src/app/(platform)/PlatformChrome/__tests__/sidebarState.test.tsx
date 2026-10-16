import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Sidebar, useSidebar } from "@/components/ui/sidebar";
import { PlatformChrome } from "../PlatformChrome";

let pathname = "/home";
let isBrainDumpEnabled = false;

vi.mock("next/navigation", () => ({
  usePathname: () => pathname,
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

vi.mock("../usePlatformChrome", () => ({
  usePlatformChrome: () => ({
    showNewLayout: true,
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
    useGetFlag: (flag: string) =>
      flag === actual.Flag.ONBOARDING_BRAIN_DUMP ? isBrainDumpEnabled : false,
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
  const view = render(
    <PlatformChrome persistedSidebarOpen={persistedSidebarOpen}>
      <StateLog />
    </PlatformChrome>,
  );
  return { ...view, paintedStates };
}

function sidebarState() {
  return document
    .querySelector("[data-collapsible]")
    ?.getAttribute("data-state");
}

function rememberSidebar(open: boolean) {
  document.cookie = `sidebar_state=${open}; path=/`;
}

beforeEach(() => {
  pathname = "/home";
  isBrainDumpEnabled = false;
  document.cookie = "sidebar_state=; path=/; max-age=0";
});

afterEach(() => {
  Object.defineProperty(window, "innerWidth", {
    value: DESKTOP_WIDTH,
    configurable: true,
  });
});

describe("the app sidebar across reloads", () => {
  it.each([
    { cohort: "brain-dump", brainDump: true, toggledTo: "expanded" },
    { cohort: "default", brainDump: false, toggledTo: "collapsed" },
  ])(
    "keeps the $cohort cohort's toggle after a reload",
    ({ brainDump, toggledTo }) => {
      isBrainDumpEnabled = brainDump;
      const first = load();
      fireEvent.click(screen.getByRole("button", { name: "Toggle Sidebar" }));
      expect(sidebarState()).toBe(toggledTo);
      first.unmount();

      load();

      expect(sidebarState()).toBe(toggledTo);
    },
  );

  it.each([
    { cohort: "brain-dump", brainDump: true, expected: "collapsed" },
    { cohort: "default", brainDump: false, expected: "expanded" },
  ])(
    "starts the $cohort cohort $expected when nothing is remembered",
    ({ brainDump, expected }) => {
      isBrainDumpEnabled = brainDump;

      load();

      expect(sidebarState()).toBe(expected);
    },
  );

  it("falls back to the cookie the server read when the browser has none", () => {
    isBrainDumpEnabled = true;

    load(true);

    expect(sidebarState()).toBe("expanded");
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
    isBrainDumpEnabled = true;
    rememberSidebar(true);
    pathname = "/build";
    load().unmount();

    pathname = "/home";
    load();

    expect(sidebarState()).toBe("expanded");
  });

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
