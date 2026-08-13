import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { render, screen } from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppSidebar } from "../AppSidebar";

// The global next/link mock only exports `default`; AppSidebar also imports
// `useLinkStatus`, so re-mock here with a no-op pending status.
vi.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
    ...props
  }: {
    children: ReactNode;
    href: string;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

// Flag-keyed map so each test controls HIRE_EXPERTS and ONBOARDING_BRAIN_DUMP
// independently — an unkeyed boolean would keep these tests green even if the
// nav were gated on the wrong flag.
const flagValues = vi.hoisted(() => ({ map: new Map<string, boolean>() }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flagValues.map.get(flag) ?? false,
  };
});

function renderSidebar() {
  return render(
    <SidebarProvider>
      <AppSidebar />
    </SidebarProvider>,
  );
}

beforeEach(() => {
  flagValues.map.clear();
  server.use(getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }));
});

afterEach(() => {
  server.resetHandlers();
});

describe("AppSidebar", () => {
  it("renders the primary navigation links", () => {
    renderSidebar();
    expect(screen.getByText("Agents")).toBeDefined();
    expect(screen.getByText("Marketplace")).toBeDefined();
    expect(screen.getByText("Build")).toBeDefined();
    expect(screen.getByText("Files")).toBeDefined();
    // /home 404s without the experts flag, so it must not be offered here.
    expect(screen.queryByText("Home")).toBeNull();
  });

  it("flag-off: omits both the Home entry and the Team row (pre-experts nav)", () => {
    renderSidebar();
    // Both /home and /team 404 without hire-experts, so neither may be offered.
    expect(screen.queryByText("Home")).toBeNull();
    expect(screen.queryByText("Team")).toBeNull();
    // The pre-experts workspace root ("Agents") stays in their place.
    expect(screen.getByText("Agents")).toBeDefined();
  });

  it("flag-off: keeps experts nav hidden when only other flags are on (HIRE_EXPERTS=false, ONBOARDING_BRAIN_DUMP=true)", () => {
    flagValues.map.set(Flag.ONBOARDING_BRAIN_DUMP, true);
    renderSidebar();
    expect(screen.queryByText("Home")).toBeNull();
    expect(screen.queryByText("Team")).toBeNull();
    expect(screen.getByText("Agents")).toBeDefined();
  });

  it("shows Team instead of Agents when the hire-experts flag is on", () => {
    flagValues.map.set(Flag.HIRE_EXPERTS, true);
    renderSidebar();
    const teamLink = screen.getByRole("link", { name: /team/i });
    expect(teamLink.getAttribute("href")).toBe("/team");
    expect(screen.queryByText("Agents")).toBeNull();
  });

  it("adds a Home link when only the hire-experts flag is on", () => {
    flagValues.map.set(Flag.HIRE_EXPERTS, true);
    renderSidebar();
    const homeLink = screen.getByRole("link", { name: /home/i });
    expect(homeLink.getAttribute("href")).toBe("/home");
  });

  it("renders the New Task call-to-action pointing at /copilot", () => {
    renderSidebar();
    const newTask = screen.getByRole("link", { name: /new task/i });
    expect(newTask.getAttribute("href")).toBe("/copilot");
  });

  it("renders the workspace and recent chats group headers", () => {
    renderSidebar();
    expect(screen.getByText("Workspace")).toBeDefined();
    expect(screen.getByText("Recent chats")).toBeDefined();
  });

  it("marks the active link based on the current pathname", () => {
    // global next/navigation mock resolves usePathname() to "/marketplace"
    renderSidebar();
    const marketplaceLink = screen.getByText("Marketplace").closest("a");
    expect(marketplaceLink?.getAttribute("href")).toBe("/marketplace");
  });

  it("shows the recent-chats empty state once sessions resolve", async () => {
    renderSidebar();
    expect(await screen.findByText(/no conversations yet/i)).toBeDefined();
  });
});
