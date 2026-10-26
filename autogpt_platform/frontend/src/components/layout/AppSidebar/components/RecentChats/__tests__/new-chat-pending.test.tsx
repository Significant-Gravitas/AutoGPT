import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RecentChats } from "../RecentChats";

// Every link reports a navigation in flight, as it does right after a click.
vi.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
    prefetch: _prefetch,
    ...props
  }: {
    children: ReactNode;
    href: string;
    prefetch?: boolean;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: true }),
}));

afterEach(() => {
  server.resetHandlers();
});

describe("RecentChats — new chat link while navigating", () => {
  it("swaps the + for a spinner while the new chat opens", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          {
            id: "autopilot-1",
            title: "Otto chat",
            is_processing: false,
            created_at: "2026-06-30T10:00:00",
            updated_at: "2026-06-30T10:00:00",
            expert_id: null,
          },
        ],
        total: 1,
      }),
      getListExpertIdentitiesMockHandler([]),
    );
    render(
      <SidebarProvider>
        <RecentChats />
      </SidebarProvider>,
    );

    const link = await screen.findByRole("link", {
      name: "New chat with Otto",
    });
    expect(link.querySelector("svg.animate-spin")).not.toBeNull();
  });
});
