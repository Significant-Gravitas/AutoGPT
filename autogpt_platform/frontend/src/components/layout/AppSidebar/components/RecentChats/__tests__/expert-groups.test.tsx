import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import { RecentChats } from "../RecentChats";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "hire-experts",
  };
});

const mariaExpert: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: "https://example.com/maria.png",
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

function makeSession(args: { id: string; title: string; expertId?: string }) {
  return {
    id: args.id,
    title: args.title,
    is_processing: false,
    created_at: "2026-06-30T10:00:00",
    updated_at: "2026-06-30T10:00:00",
    expert_id: args.expertId ?? null,
  };
}

function makeSessions(count: number, expertId?: string) {
  const prefix = expertId ?? "autopilot";
  return Array.from({ length: count }, (_, index) =>
    makeSession({
      id: `${prefix}-${index + 1}`,
      title: `${prefix} chat ${index + 1}`,
      expertId,
    }),
  );
}

function renderRecentChats() {
  return render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

afterEach(() => {
  server.resetHandlers();
});

describe("RecentChats — expert groups", () => {
  it("collapses and expands a group when its header is clicked", async () => {
    const sessions = makeSessions(2);
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertsMockHandler([]),
    );
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 1")).toBeDefined();

    const header = screen.getByRole("button", { name: "Autopilot chats" });
    fireEvent.click(header);
    expect(screen.queryByText("autopilot chat 1")).toBeNull();

    fireEvent.click(header);
    expect(await screen.findByText("autopilot chat 1")).toBeDefined();
  });

  it("shows only the first 6 chats and reveals more via Load more", async () => {
    const sessions = makeSessions(14);
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertsMockHandler([]),
    );
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 6")).toBeDefined();
    expect(screen.queryByText("autopilot chat 7")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /load more/i }));
    expect(await screen.findByText("autopilot chat 12")).toBeDefined();
    expect(screen.queryByText("autopilot chat 13")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /load more/i }));
    expect(await screen.findByText("autopilot chat 14")).toBeDefined();
    expect(screen.queryByRole("button", { name: /load more/i })).toBeNull();
  });

  it("groups expert chats under the expert's name with their own preview", async () => {
    const sessions = [...makeSessions(2), ...makeSessions(7, mariaExpert.id)];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertsMockHandler([mariaExpert]),
    );
    renderRecentChats();

    expect(
      await screen.findByRole("button", { name: "Maria chats" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Autopilot chats" }),
    ).toBeDefined();

    expect(screen.getByText("autopilot chat 2")).toBeDefined();
    expect(screen.getByText("expert-maria chat 6")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 7")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /load more/i }));
    expect(await screen.findByText("expert-maria chat 7")).toBeDefined();
  });
});
