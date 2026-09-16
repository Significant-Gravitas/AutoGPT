import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { SESSION_LIST_REFETCH_INTERVAL_MS } from "@/app/(platform)/copilot/useSessionList";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [
    "The expert discloses that it is AI when acting externally.",
    "External actions require approval.",
  ],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

function makeSession(args: {
  id: string;
  title: string;
  expertId?: string;
  isProcessing?: boolean;
}) {
  return {
    id: args.id,
    title: args.title,
    is_processing: args.isProcessing ?? false,
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

function groupHeader(label: string) {
  return screen.getByRole("button", { name: `${label} chats` });
}

/** Groups render open, so this is how a test gets to the collapsed state. */
async function collapseGroup(label: string) {
  fireEvent.click(
    await screen.findByRole("button", { name: `${label} chats` }),
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
});

describe("RecentChats — expert groups", () => {
  it("starts open and collapses when its header is clicked", async () => {
    const sessions = makeSessions(2);
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 1")).toBeDefined();

    fireEvent.click(groupHeader("Autopilot"));
    expect(screen.queryByText("autopilot chat 1")).toBeNull();

    fireEvent.click(groupHeader("Autopilot"));
    expect(await screen.findByText("autopilot chat 1")).toBeDefined();
  });

  it("keeps a running chat visible while the group is collapsed", async () => {
    const sessions = [
      makeSession({ id: "running", title: "running chat", isProcessing: true }),
      ...makeSessions(2),
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();
    await collapseGroup("Autopilot");

    expect(await screen.findByText("running chat")).toBeDefined();
    expect(screen.queryByText("autopilot chat 1")).toBeNull();
  });

  it("shows only the first 10 chats and reveals more via Load more", async () => {
    const sessions = makeSessions(22);
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    expect(await screen.findByText("autopilot chat 10")).toBeDefined();
    expect(screen.queryByText("autopilot chat 11")).toBeNull();

    const loadMore = () =>
      screen.getByRole("button", { name: "Load more Autopilot chats" });

    fireEvent.click(loadMore());
    expect(await screen.findByText("autopilot chat 20")).toBeDefined();
    expect(screen.queryByText("autopilot chat 21")).toBeNull();

    fireEvent.click(loadMore());
    expect(await screen.findByText("autopilot chat 22")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Load more Autopilot chats" }),
    ).toBeNull();
  });

  it("groups expert chats under the expert's name with independent previews", async () => {
    const sessions = [...makeSessions(12), ...makeSessions(11, mariaExpert.id)];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([mariaExpert]),
    );
    renderRecentChats();

    expect(await screen.findByText("expert-maria chat 10")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 11")).toBeNull();
    expect(screen.queryByText("autopilot chat 11")).toBeNull();

    fireEvent.click(
      screen.getByRole("button", { name: "Load more Maria chats" }),
    );
    expect(await screen.findByText("expert-maria chat 11")).toBeDefined();
    expect(screen.queryByText("autopilot chat 11")).toBeNull();
  });

  it("falls back to a generic Expert label when the expert is unknown", async () => {
    const sessions = makeSessions(2, "expert-ghost");
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    expect(
      await screen.findByRole("button", { name: "Expert chats" }),
    ).toBeDefined();
    expect(await screen.findByText("expert-ghost chat 1")).toBeDefined();
  });

  it("keeps the group-level and list-level Load more buttons distinct", async () => {
    const sessions = makeSessions(12);
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions,
        total: sessions.length + 10,
      }),
      getListExpertIdentitiesMockHandler([]),
    );
    renderRecentChats();

    expect(
      await screen.findByRole("button", { name: "Load more Autopilot chats" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Load more" })).toBeDefined();
  });

  it("keeps reveal and collapse state across session list refetches", async () => {
    const sessions = [...makeSessions(12), ...makeSessions(2, mariaExpert.id)];
    let listCalls = 0;
    server.use(
      http.get("*/api/chat/sessions", () => {
        listCalls++;
        return HttpResponse.json({ sessions, total: sessions.length });
      }),
      getListExpertIdentitiesMockHandler([mariaExpert]),
    );
    renderRecentChats();

    await collapseGroup("Maria");
    fireEvent.click(
      await screen.findByRole("button", { name: "Load more Autopilot chats" }),
    );
    expect(await screen.findByText("autopilot chat 11")).toBeDefined();

    const callsBefore = listCalls;
    vi.advanceTimersByTime(SESSION_LIST_REFETCH_INTERVAL_MS);
    await waitFor(() => expect(listCalls).toBeGreaterThan(callsBefore));

    expect(screen.getByText("autopilot chat 11")).toBeDefined();
    expect(screen.queryByText("expert-maria chat 1")).toBeNull();
  });
});
