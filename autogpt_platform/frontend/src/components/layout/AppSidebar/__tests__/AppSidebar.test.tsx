import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetHomeDashboardMockHandler200,
  getGetHomeDashboardMockHandler401,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import type { HomeAgentStatusStatus } from "@/app/api/__generated__/models/homeAgentStatusStatus";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppSidebar } from "../AppSidebar";
import { SIDEBAR_TEAM_PREVIEW_COUNT } from "../components/SidebarTeamMembers/SidebarTeamMembers";

function dashboardWith(agents: HomeAgentStatus[]): HomeDashboardResponse {
  return { ...getGetHomeDashboardResponseMock200(), agents };
}

function makeAgent(
  id: string,
  name: string,
  status: HomeAgentStatusStatus,
): HomeAgentStatus {
  return {
    expert: { id, name, role: "Expert", avatar_url: null },
    status,
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
  server.use(
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
    getGetHomeDashboardMockHandler200(dashboardWith([])),
  );
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

  it("nests hired experts under Team with a Your AI row and a Hire link", async () => {
    useGetFlagMock.mockReturnValue(true);
    server.use(
      getGetHomeDashboardMockHandler200(
        dashboardWith([makeAgent("expert-maria", "Maria", "working")]),
      ),
    );
    renderSidebar();

    const memberLink = await screen.findByRole("link", { name: /Maria/i });
    expect(memberLink.getAttribute("href")).toBe(
      "/copilot?expertId=expert-maria",
    );

    const yourAi = screen.getByRole("link", { name: /your ai/i });
    expect(yourAi.getAttribute("href")).toBe("/copilot");

    const hire = screen.getByRole("link", { name: /^hire$/i });
    expect(hire.getAttribute("href")).toBe("/marketplace#experts");
  });

  it("maps each member status to its presence colour", async () => {
    useGetFlagMock.mockReturnValue(true);
    server.use(
      getGetHomeDashboardMockHandler200(
        dashboardWith([
          makeAgent("e-working", "Working Expert", "working"),
          makeAgent("e-ready", "Ready Expert", "ready"),
          makeAgent("e-paused", "Paused Expert", "paused"),
          makeAgent("e-setup", "Setup Expert", "needs_setup"),
          makeAgent("e-failed", "Failed Expert", "failed"),
        ]),
      ),
    );
    renderSidebar();

    expect(
      (await screen.findByRole("img", { name: "Working" })).className,
    ).toContain("bg-amber-500");
    expect(screen.getByRole("img", { name: "Ready" }).className).toContain(
      "bg-emerald-500",
    );
    expect(screen.getByRole("img", { name: "Paused" }).className).toContain(
      "bg-zinc-300",
    );
    expect(
      screen.getByRole("img", { name: "Needs setup" }).className,
    ).toContain("bg-zinc-300");
    expect(
      screen.getByRole("img", { name: "Needs attention" }).className,
    ).toContain("bg-red-500");
  });

  it("keeps Your AI and Hire visible when the user has no hired experts", async () => {
    useGetFlagMock.mockReturnValue(true);
    renderSidebar();

    const yourAi = await screen.findByRole("link", { name: /your ai/i });
    expect(yourAi.getAttribute("href")).toBe("/copilot");
    expect(
      screen.getByRole("link", { name: /^hire$/i }).getAttribute("href"),
    ).toBe("/marketplace#experts");
  });

  it("keeps Your AI and Hire visible when the dashboard request fails", async () => {
    useGetFlagMock.mockReturnValue(true);
    server.use(getGetHomeDashboardMockHandler401());
    const dashboardRequestSettled = new Promise<void>((resolve) => {
      function onResponse({ request }: { request: Request }) {
        if (new URL(request.url).pathname !== "/api/proxy/api/home") return;
        server.events.removeListener("response:mocked", onResponse);
        resolve();
      }

      server.events.on("response:mocked", onResponse);
    });
    renderSidebar();

    await dashboardRequestSettled;

    const yourAi = screen.getByRole("link", { name: /your ai/i });
    expect(yourAi.getAttribute("href")).toBe("/copilot");
    expect(
      screen.getByRole("link", { name: /^hire$/i }).getAttribute("href"),
    ).toBe("/marketplace#experts");
  });

  it("caps the member list and keeps Hire reachable via View all", async () => {
    useGetFlagMock.mockReturnValue(true);
    server.use(
      getGetHomeDashboardMockHandler200(
        dashboardWith(
          Array.from({ length: 8 }, (_, i) =>
            makeAgent(`expert-${i}`, `Expert ${i}`, "ready"),
          ),
        ),
      ),
    );
    renderSidebar();

    expect(await screen.findByRole("link", { name: /Expert 4/ })).toBeDefined();
    expect(screen.queryByRole("link", { name: /Expert 5/ })).toBeNull();

    const viewAll = screen.getByRole("link", { name: /view all \(8\)/i });
    expect(viewAll.getAttribute("href")).toBe("/team");

    expect(
      screen.getByRole("link", { name: /^hire$/i }).getAttribute("href"),
    ).toBe("/marketplace#experts");
  });

  it("shows every member without View all at exactly the preview count", async () => {
    useGetFlagMock.mockReturnValue(true);
    server.use(
      getGetHomeDashboardMockHandler200(
        dashboardWith(
          Array.from({ length: SIDEBAR_TEAM_PREVIEW_COUNT }, (_, i) =>
            makeAgent(`expert-${i}`, `Expert ${i}`, "ready"),
          ),
        ),
      ),
    );
    renderSidebar();

    expect(
      await screen.findByRole("link", {
        name: new RegExp(`Expert ${SIDEBAR_TEAM_PREVIEW_COUNT - 1}`),
      }),
    ).toBeDefined();
    expect(screen.queryByRole("link", { name: /view all/i })).toBeNull();
  });

  it("shows View all as soon as the roster exceeds the preview count", async () => {
    useGetFlagMock.mockReturnValue(true);
    server.use(
      getGetHomeDashboardMockHandler200(
        dashboardWith(
          Array.from({ length: SIDEBAR_TEAM_PREVIEW_COUNT + 1 }, (_, i) =>
            makeAgent(`expert-${i}`, `Expert ${i}`, "ready"),
          ),
        ),
      ),
    );
    renderSidebar();

    const viewAll = await screen.findByRole("link", {
      name: new RegExp(`view all \\(${SIDEBAR_TEAM_PREVIEW_COUNT + 1}\\)`, "i"),
    });
    expect(viewAll.getAttribute("href")).toBe("/team");
    expect(
      screen.queryByRole("link", {
        name: new RegExp(`Expert ${SIDEBAR_TEAM_PREVIEW_COUNT}`),
      }),
    ).toBeNull();
  });

  it("omits the nested team members when the hire-experts flag is off", () => {
    renderSidebar();
    expect(screen.queryByText("Your AI")).toBeNull();
    expect(screen.queryByRole("link", { name: /^hire$/i })).toBeNull();
  });
});
