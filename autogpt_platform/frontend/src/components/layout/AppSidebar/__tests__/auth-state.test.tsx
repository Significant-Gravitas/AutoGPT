import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import { act, render, screen } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppSidebar } from "../AppSidebar";

const mockUseAuth = vi.fn();
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => mockUseAuth(),
}));

vi.mock("next/link", () => ({
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

vi.mock("../components/SidebarUserActions/SidebarUserActions", () => ({
  SidebarUserActions: () => null,
}));

const sessionsRequest = vi.fn(() =>
  HttpResponse.json({
    sessions: [
      {
        id: "session-1",
        title: "My real conversation",
        is_processing: false,
        created_at: "2026-06-30T00:00:00Z",
        updated_at: "2026-06-30T00:00:00Z",
      },
    ],
    total: 1,
  }),
);

function renderSidebar() {
  return render(
    <SidebarProvider>
      <AppSidebar />
    </SidebarProvider>,
  );
}

beforeEach(() => {
  sessionsRequest.mockClear();
  server.use(http.get("*/api/chat/sessions", sessionsRequest));
});

describe("AppSidebar auth state", () => {
  it.each([false, true])(
    "hides recent chats and does not fetch sessions without a user (loading %s)",
    async (isUserLoading) => {
      mockUseAuth.mockReturnValue({
        user: null,
        isLoggedIn: false,
        isUserLoading,
      });

      await act(async () => {
        renderSidebar();
      });

      expect(screen.queryByText("Recent chats")).toBeNull();
      expect(screen.queryByText("My real conversation")).toBeNull();
      expect(screen.queryByText(/no conversations yet/i)).toBeNull();
      expect(sessionsRequest).not.toHaveBeenCalled();
    },
  );

  it("fetches and renders the signed-in user's recent chats", async () => {
    mockUseAuth.mockReturnValue({
      user: { id: "user-1", email: "alice@example.com", role: "user" },
      isLoggedIn: true,
      isUserLoading: false,
    });

    renderSidebar();

    expect(await screen.findByText("My real conversation")).toBeDefined();
    expect(screen.getByText("Recent chats")).toBeDefined();
    expect(sessionsRequest).toHaveBeenCalledOnce();
  });
});
