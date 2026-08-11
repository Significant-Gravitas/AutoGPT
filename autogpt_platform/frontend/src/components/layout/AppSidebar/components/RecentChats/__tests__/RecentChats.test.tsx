import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { SidebarProvider } from "@/components/ui/sidebar";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { RecentChats } from "../RecentChats";

const mockEnabledFlags = new Set<string>(["chat-sharing"]);

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => mockEnabledFlags.has(flag),
  };
});

interface SessionSeed {
  id: string;
  title: string;
  updated_at: string;
  is_pinned?: boolean;
}

function makeSession(seed: SessionSeed) {
  return {
    id: seed.id,
    title: seed.title,
    is_processing: false,
    is_pinned: seed.is_pinned ?? false,
    created_at: seed.updated_at,
    updated_at: seed.updated_at,
  };
}

const TODAY = "2026-06-30T10:00:00";
const YESTERDAY = "2026-06-29T10:00:00";

function renderRecentChats() {
  return render(
    <SidebarProvider>
      <RecentChats />
    </SidebarProvider>,
  );
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  vi.setSystemTime(new Date("2026-06-30T12:00:00"));
  useCopilotUIStore.setState({ sessionToDelete: null });
});

afterEach(() => {
  vi.useRealTimers();
  server.resetHandlers();
  mockEnabledFlags.delete("chat-pinning");
});

describe("RecentChats — empty + loading", () => {
  it("shows the empty state when there are no sessions", async () => {
    server.use(getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }));
    renderRecentChats();
    expect(await screen.findByText(/no conversations yet/i)).toBeDefined();
  });
});

describe("RecentChats — grouped list", () => {
  it("renders sessions grouped under date labels", async () => {
    const sessions = [
      makeSession({ id: "a", title: "Today chat", updated_at: TODAY }),
      makeSession({
        id: "b",
        title: "Yesterday chat",
        updated_at: YESTERDAY,
      }),
    ];
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
    );
    renderRecentChats();

    expect(await screen.findByText("Today chat")).toBeDefined();
    expect(screen.getByText("Yesterday chat")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("Yesterday")).toBeDefined();
  });
});

describe("RecentChats — pinned split", () => {
  const pinnedAndDated = [
    makeSession({
      id: "pinned",
      title: "Pinned chat",
      updated_at: YESTERDAY,
      is_pinned: true,
    }),
    makeSession({ id: "loose", title: "Loose chat", updated_at: TODAY }),
  ];

  it("lifts pinned sessions out of the date groups", async () => {
    mockEnabledFlags.add("chat-pinning");
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: pinnedAndDated,
        total: pinnedAndDated.length,
      }),
    );
    renderRecentChats();

    const pinnedGroup = (await screen.findByText("Pinned")).closest(
      "div",
    )?.parentElement;
    if (!pinnedGroup) throw new Error("pinned group not found");
    expect(within(pinnedGroup).getByText("Pinned chat")).toBeDefined();

    // The only "Yesterday" session is pinned, so that date group disappears.
    expect(screen.queryByText("Yesterday")).toBeNull();

    const todayGroup = screen.getByText("Today").closest("div")?.parentElement;
    if (!todayGroup) throw new Error("today group not found");
    expect(within(todayGroup).getByText("Loose chat")).toBeDefined();
    expect(within(todayGroup).queryByText("Pinned chat")).toBeNull();
  });

  it("keeps pinned sessions in their date group when the flag is off", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: pinnedAndDated,
        total: pinnedAndDated.length,
      }),
    );
    renderRecentChats();

    await screen.findByText("Pinned chat");
    expect(screen.queryByText("Pinned")).toBeNull();
    const yesterdayGroup = screen
      .getByText("Yesterday")
      .closest("div")?.parentElement;
    if (!yesterdayGroup) throw new Error("yesterday group not found");
    expect(within(yesterdayGroup).getByText("Pinned chat")).toBeDefined();
  });
});

describe("RecentChats — load more", () => {
  it("hides 'Load more' when all sessions are loaded", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [makeSession({ id: "a", title: "Only", updated_at: TODAY })],
        total: 1,
      }),
    );
    renderRecentChats();
    await screen.findByText("Only");
    expect(screen.queryByRole("button", { name: /load more/i })).toBeNull();
  });

  it("shows 'Load more' and fetches the next page", async () => {
    const seenOffsets: string[] = [];
    server.use(
      http.get("*/api/chat/sessions", ({ request }) => {
        const offset = new URL(request.url).searchParams.get("offset") ?? "0";
        seenOffsets.push(offset);
        const sessions =
          Number(offset) === 0
            ? [makeSession({ id: "p0", title: "Page 0", updated_at: TODAY })]
            : [makeSession({ id: "p1", title: "Page 1", updated_at: TODAY })];
        return HttpResponse.json({ sessions, total: 2 });
      }),
    );
    renderRecentChats();

    const loadMore = await screen.findByRole("button", { name: /load more/i });
    fireEvent.click(loadMore);

    expect(await screen.findByText("Page 1")).toBeDefined();
    expect(seenOffsets).toContain("1");
  });
});

describe("RecentChats — delete flow", () => {
  it("stages a session for deletion when Delete is chosen", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          makeSession({ id: "x", title: "Deletable", updated_at: TODAY }),
        ],
        total: 1,
      }),
    );
    renderRecentChats();

    const row = (await screen.findByText("Deletable")).closest("li");
    if (!row) throw new Error("row not found");
    fireEvent.pointerDown(
      within(row).getByRole("button", { name: /chat actions/i }),
      { button: 0 },
    );
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /delete chat/i }),
    );

    expect(useCopilotUIStore.getState().sessionToDelete).toMatchObject({
      id: "x",
    });
  });
});
