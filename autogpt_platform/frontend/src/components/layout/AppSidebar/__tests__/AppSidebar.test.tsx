import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetHomeDashboardMockHandler200,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppSidebar } from "../AppSidebar";

function dashboardWith(agents: HomeAgentStatus[]): HomeDashboardResponse {
  return { ...getGetHomeDashboardResponseMock200(), agents };
}

function makeAgent(id: string, name: string): HomeAgentStatus {
  return {
    expert: { id, name, role: "Expert", avatar_url: null },
    status: "ready",
    detail: "Ready for the next task",
  };
}

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

const useGetFlagMock = vi.hoisted(() => vi.fn(() => false));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => useGetFlagMock(),
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
  useGetFlagMock.mockReturnValue(false);
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

  it("shows Team instead of Agents when the hire-experts flag is on", () => {
    useGetFlagMock.mockReturnValue(true);
    renderSidebar();
    const teamLink = screen.getByRole("link", { name: /team/i });
    expect(teamLink.getAttribute("href")).toBe("/team");
    expect(screen.queryByText("Agents")).toBeNull();
  });

  it("adds a Home link when the hire-experts flag is on", () => {
    useGetFlagMock.mockReturnValue(true);
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

  it("does not nest the team roster under Team", async () => {
    useGetFlagMock.mockReturnValue(true);
    server.use(
      getGetHomeDashboardMockHandler200(
        dashboardWith([makeAgent("expert-maria", "Maria")]),
      ),
    );
    renderSidebar();

    expect(await screen.findByRole("link", { name: /team/i })).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByRole("link", { name: /Maria/i })).toBeNull(),
    );
  });
});
