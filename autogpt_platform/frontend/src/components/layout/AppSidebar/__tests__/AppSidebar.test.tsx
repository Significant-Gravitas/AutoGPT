import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
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

function renderSidebar() {
  return render(
    <SidebarProvider>
      <AppSidebar />
    </SidebarProvider>,
  );
}

beforeEach(() => {
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
