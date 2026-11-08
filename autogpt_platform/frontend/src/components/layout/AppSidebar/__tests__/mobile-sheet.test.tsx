import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  SidebarProvider,
  SidebarTrigger,
} from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AppSidebar } from "../AppSidebar";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "alice@example.com", role: "user" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

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

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
    usePathname: () => "/home",
    useSearchParams: () => new URLSearchParams(),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

vi.mock("@/hooks/use-mobile", () => ({ useIsMobile: () => true }));

function renderMobileSidebar() {
  render(
    <SidebarProvider>
      <SidebarTrigger />
      <AppSidebar />
    </SidebarProvider>,
  );
}

function openSheet() {
  fireEvent.click(screen.getByRole("button", { name: /toggle sidebar/i }));
  return screen.getByRole("dialog");
}

beforeEach(() => {
  server.use(
    getGetV2ListSessionsMockHandler200({
      sessions: [
        {
          id: "session-1",
          title: "Plan the launch",
          is_processing: false,
          is_pinned: false,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ],
      total: 1,
    }),
  );
});

afterEach(() => {
  server.resetHandlers();
});

describe("AppSidebar on mobile", () => {
  it("closes the sheet when a chat is tapped", async () => {
    renderMobileSidebar();
    openSheet();

    fireEvent.click(
      await screen.findByRole("link", { name: /plan the launch/i }),
    );

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("closes the sheet when a nav link is tapped", async () => {
    renderMobileSidebar();
    openSheet();

    fireEvent.click(screen.getByRole("link", { name: /marketplace/i }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("keeps the sheet open when a tap opens a new tab", async () => {
    renderMobileSidebar();
    openSheet();

    fireEvent.click(screen.getByRole("link", { name: /marketplace/i }), {
      metaKey: true,
    });

    expect(screen.getByRole("dialog")).toBeDefined();
  });

  it("keeps the sheet open when a chat's actions menu opens", async () => {
    renderMobileSidebar();
    openSheet();

    await screen.findByRole("link", { name: /plan the launch/i });
    fireEvent.click(screen.getByRole("button", { name: /chat actions/i }));

    expect(screen.getByRole("dialog")).toBeDefined();
  });
});
